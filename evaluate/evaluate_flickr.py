import os
import json
import tempfile
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def evaluate_flickr(results, gt_file=None):
    """
    Evaluate Flickr captioning results using pycocoevalcap metrics.
    Compatible with outputs formatted like:
        {
            "image": [...],             # list of image paths or IDs
            "final_caption": [...],     # list of generated captions
            "captions_gt": [...]        # list of reference captions
        }

    Returns a dict of BLEU, METEOR, ROUGE_L, CIDEr, SPICE scores.
    """

    # --- Build COCO-style GT and predictions ---
    annotations, predictions = [], []

    for idx, item in enumerate(results):
        image_ids = item["image"]
        preds = item.get("final_caption", item.get("caption"))
        refs = item.get("captions_gt", item.get("captions"))

        # Normalize list formats
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(refs, str):
            refs = [refs]

        for i, (img_path, pred, ref) in enumerate(zip(image_ids, preds, refs)):
            image_id = idx * 1000 + i
            annotations.append({
                "image_id": image_id,
                "id": image_id,
                "caption": ref
            })
            predictions.append({
                "image_id": image_id,
                "caption": pred
            })

    # --- Save temporary COCO-style files ---
    tmp_dir = tempfile.mkdtemp()
    ref_path = os.path.join(tmp_dir, "flickr_refs.json")
    pred_path = os.path.join(tmp_dir, "flickr_preds.json")

    coco_data = {
        "info": {"description": "Flickr-style caption evaluation"},
        "images": [{"id": ann["image_id"]} for ann in annotations],
        "annotations": annotations,
        "licenses": [],
    }

    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # --- Evaluate with pycocoevalcap ---
    coco = COCO(ref_path)
    cocoRes = coco.loadRes(pred_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    scores = {metric: score for metric, score in cocoEval.eval.items()}

    print("\n========== Flickr Caption Evaluation ==========")
    for k, v in scores.items():
        print(f"{k:<10}: {v:.4f}")
    print("==============================================\n")

    return scores
