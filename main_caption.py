import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from transformers.utils import logging
logging.set_verbosity_error()
import argparse
import os
import json
import re
import torch
torch.backends.cudnn.enabled = False
from tqdm import tqdm

from parser import update_args

from models.vlm.getter import vlm_getter
from models.experts.getter import get_experts
from models.distance_model import get_dist_model

from evaluate.evaluate_cococaption import evaluate_cococaption
from evaluate.evaluate_nocaps import evaluate_nocaps
from dataset.getter import get_caption_loader


COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

grid_hw = {
    'idefics': (224, 224),
    'pali': (448, 448),
    'instructblip': (224, 224),
    'llava': (336, 336),
    'qwen_vl': (None, None),
    'cogvlm': (336, 336),
    'minigptv2': (448, 448),
    'llama32': (336, 336),
}

evaluate_fn = {
    'cococaption': evaluate_cococaption,
    'nocaps': evaluate_nocaps,
    'flickr': evaluate_cococaption,
}

from process.mask import (
    generate_gumbel_topk_masks, 
    generate_gumbel_bottomk_masks,
    generate_dense_topk_masks,
    generate_dense_bottomk_masks,
    generate_gumbel_hybrid_masks
)
mask_fns = {
    'topk': generate_gumbel_topk_masks,
    'bottomk': generate_gumbel_bottomk_masks,
    'dense_topk': generate_dense_topk_masks,
    'dense_bottomk': generate_dense_bottomk_masks,
    'hybrid': generate_gumbel_hybrid_masks,
}
from process.uq_caption import (
    compute_uq_caption_faithfulness, 
    compute_uq_caption_fidelity,
    compute_uq_caption_hybrid,
)
compute_uq_fns = {
    'relevance_faithfulness': compute_uq_caption_faithfulness,
    'relevance_fidelity': compute_uq_caption_fidelity,
    'hybrid': compute_uq_caption_hybrid,
}


def main(args):    
    print(f"Arguments: {args}")

    with open(args.data_path, 'r') as f:
        json_file = json.load(f)
        data_paths = json_file["data_paths"]
        eval_paths = json_file["eval_paths"]

    print(f"Loading VLM: {args.vlm} on {args.vlm_device}")
    vlm, vlm_processor, infer_vlm = vlm_getter(
        args.vlm, args.vlm_device, args.dataset, cache_dir=args.cache_dir)

    val_loader = get_caption_loader(
        args,
        data_path=data_paths[args.dataset],
        split=args.split,
        batch_size=args.batch_size,
        num_workers=0
    )

    if args.dnr:
        print(f"Loading Experts: {args.experts} on {args.expert_device}")
        experts = get_experts(args, args.expert_device)

        print(f"Loading Distance Model: {args.dist_model} on {args.dist_device}")
        dist_model, dist_tokenizer = get_dist_model(args.dist_model, args.dist_device)

        mask_fn = mask_fns[args.mask_func]
        compute_uq = compute_uq_fns[args.uq_func]
        save_dir = f"results/{args.dataset}/{args.vlm}"
        os.makedirs(save_dir, exist_ok=True)

    results, num = [], 0
    for batch in tqdm(val_loader):
        num += len(batch['images'])
        drafts = infer_vlm(vlm, vlm_processor, batch=batch)
        
        if args.dnr:
            H, W = grid_hw[args.vlm]
            batch["questions"] = ["describe the image"] * len(batch["images"])
            batch["query"] = [COCO_CATEGORIES] * len(batch["images"])

            r_map_paths = []
            for img_path in batch["images"]:
                path = img_path.replace("images", "clipseg_maps")
                path = re.sub(r"\.(jpg|png)$", "", path, flags=re.IGNORECASE) 
                final_path = f"{path}_objects.pt"
                r_map_paths.append(final_path)
            batch["r_map"] = r_map_paths
    
            masks = mask_fn(
                batch['r_map'], (336, 336),
                num_masks=args.uq_num_masks,
                area_ratio=args.uq_area_ratio,
                beta=args.uq_beta,
                device=args.expert_device
            )

            prev_Uqs = compute_uq(
                drafts, masks, (H, W),
                vlm, vlm_processor, infer_vlm,
                dist_model, 
                batch, alpha=args.uq_alpha, args=args
            )
            
            all_refines, all_Uqs = {}, {}
            for key, expert in experts.items():
                expert_model = expert['model']
                expert_processor = expert['processor']
                expert_run = expert['run']
    
                expert_imgs, expert_context = expert_run(
                    model=expert_model,
                    processor=expert_processor,
                    images=batch['images'],
                    prompts=batch['questions'],
                    queries=batch['query'],
                    device=args.expert_device,
                    mode=args.rendering_mode,
                )
                expert_batch = batch.copy()
                expert_batch['images'] = expert_imgs
                expert_batch['drafts'] = drafts
                    
                refines = infer_vlm(vlm, vlm_processor, expert_batch)

                Uqs = compute_uq(
                    drafts, masks, (H, W), 
                    vlm, vlm_processor, infer_vlm,
                    dist_model, 
                    expert_batch, alpha=args.uq_alpha, args=args   
                )
                all_refines[key] = refines
                all_Uqs[key] = Uqs
                
        if args.dnr:
            captions = []
            for i, draft in enumerate(drafts):
                best_delta = 0
                chosen = draft
                for key in experts.keys():
                    delta = all_Uqs[key][i] - prev_Uqs[i]
                    if delta > best_delta: 
                        best_delta = delta
                        chosen = all_refines[key][i]
                captions.append(chosen)
        else:
            captions = drafts

        if args.dataset == 'cococaption' or args.dataset == 'flickr':
            for img_path, pred_cap, gt_caps in zip(
                batch['images'], captions, batch['captions']
            ):
                results.append({
                    'image': img_path,
                    'final_caption': pred_cap,
                    'captions_gt': gt_caps
                })
        elif args.dataset == 'nocaps':
            for img_id, img_path, pred_cap, gt_caps in zip(
                batch['image_ids'], batch['images'], captions, batch['captions']
            ):
                results.append({
                    'image_id': img_id,
                    'image': img_path,
                    'final_caption': pred_cap,
                    'captions_gt': gt_caps
                })
        else:
            raise NotImplementedError
    
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    with open(f'results/{args.dataset}/{args.vlm}_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    eval_path = eval_paths.get(args.dataset, None)
    evaluate_fn[args.dataset](results, eval_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Agent CLI")
    parser.add_argument('--cache_dir', type=str, default='/media/Data1/huggingface', help='Cache directory for models')
    parser.add_argument('--vlm', type=str, default='llava', choices=['idefics', 'pali', 'instructblip', 'llava', 'qwen_vl', 'cogvlm', 'minigptv2', 'llama32'], help='VLM model to use')
    parser.add_argument('--vlm_device', default='cuda:0', help='Device ID to use')

    parser.add_argument('--dnr', default=False, action='store_true')
    parser.add_argument('--experts', default=['grounding_dino', 'sam', 'depth_anything_v2_front', 'mdetr'], nargs='+', help='List of experts to use')
    parser.add_argument('--expert_device', type=int, default=0, help='Device ID for experts')
    parser.add_argument('--rendering_mode', default='gray', choices=['gray', 'black', 'white', 'highlight'], help='Rendering mode for experts')

    parser.add_argument('--dist_model', type=str, default='sbert', choices=['clip', 'sbert'], help='Model for computing distance in UQ')
    parser.add_argument('--dist_device', type=str, default='cuda:0', help='Device for distance model')

    parser.add_argument('--dataset', type=str, default='cococaption', choices=['cococaption', 'nocaps', 'flickr'], help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='configs/data.json', help='Path to data configuration JSON file')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--limit', type=int, default=2000, help='Limit number of samples to process')

    # ** Importance **
    parser.add_argument('--mask_func', type=str, default='hybrid', choices=['topk', 'bottomk', 'dense_topk', 'dense_bottomk', 'hybrid'])
    parser.add_argument('--uq_func', type=str, default='hybrid', choices=['relevance_faithfulness', 'relevance_fidelity', 'hybrid'])
    parser.add_argument('--uq_num_masks', type=int, default=4)
    parser.add_argument('--uq_area_ratio', type=float, default=0.25)
    parser.add_argument('--uq_alpha', type=float, default=None)
    parser.add_argument('--uq_beta', type=float, default=1.0)
    
    # Merge Expert Outputs
    parser.add_argument('--expert_gamma', type=float, default=1.0)

    args = parser.parse_args()
    update_args(args)
    main(args)