import argparse
import re
import os
import torch
import json
from tqdm import tqdm

from dataset.getter import get_qa_loader
from process.rm import build_relevance_maps


def get_loader(args, data_paths, split='val'):
    val_loader = get_qa_loader(
        args,
        data_path=data_paths[args.dataset],
        split=split,
        batch_size=args.batch_size,
        num_workers=0
    )
    return val_loader


def main(args):    
    with open(args.data_path, 'r') as f:
        data_paths = json.load(f)['data_paths']

    val_loader = get_loader(args, data_paths, split=split)

    num = 0
    for batch in tqdm(val_loader):
        queries = []
        for img_path, ques in zip(batch['images'], batch['questions']):
            que_path = img_path.replace("images", "query_simple")
            que_path = re.sub(r'\.(jpg|png)$', '', que_path, flags=re.IGNORECASE)
            safe_q = re.sub(r'[^a-zA-Z0-9]', '_', ques)
            
            if args.dataset == 'mmbench' and len(safe_q) > 150:
                safe_q = safe_q[:60] + "_TRUNC_" + safe_q[-40:]

            que_path = f"{que_path}_{safe_q}.pb"
            if not os.path.exists(que_path):
                raise FileNotFoundError(f"Query file not found: {que_path}")
            data = torch.load(que_path)
            queries.append(data["query"])
        batch['query'] = queries

        num = build_relevance_maps(
            args.dataset,
            batch['images'], batch['questions'], batch['query'],
            device=args.device, num=num,
            mmbench = True if args.dataset == 'mmbench' else False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Agent CLI")
    parser.add_argument('--dataset', type=str, default='vqav2', 
                        choices=[
                            'vqav2', 'vizwiz', 'gqa', 'textvqa', 'ocrvqa',
                            'vsr', 'vcr',
                            'okvqa', 'aokvqa', 'sqa', 
                            'mme', 'mmbench', 'seedbench', 
                            'haloquest', 'mmhalbench'
                        ], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='configs/data.json')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to process')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device ID to use')

    args = parser.parse_args()
    if args.dataset in ['vqav2', 'vizwiz', 'textvqa', 'sqa', 'okvqa', 'aokvqa', 'vcr']:
        split = 'val'
    if args.dataset in ['gqa']:
        split = 'testdev'
    if args.dataset in ['mme', 'mmbench', 'seedbench', 'haloquest', 'mmhalbench']:
        split = None
    if args.dataset in ['ocrvqa', 'vsr']:
        split = 'test'
    main(args)