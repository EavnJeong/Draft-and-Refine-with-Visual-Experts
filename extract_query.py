import argparse
import json

from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()

from dataset.getter import get_qa_loader
from models.llm.getter import llm_getter
from process.build_query import build_query


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
    val_loader = get_loader(args, data_paths, split=args.split)
    
    print(f"Arguments: {args}")
    model, tokenizer = llm_getter(args.llm, args.llm_device)

    for batch in tqdm(val_loader):
        queries = build_query(
            model, tokenizer,
            batch['images'], batch['questions'], 
            mmbench=True if args.dataset == 'mmbench' else False
        )
        for qu in queries:
            print(f"Question: {qu['question']}")
            print(f"Query: {qu['query']}")
            print('-'*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Agent CLI")
    parser.add_argument('--llm', type=str, default='llama-3-70b', choices=['llama-3-8b', 'llama-3-70b'], help='LLM model to use')
    parser.add_argument('--llm_device', default='cuda:0', help='Set auto for multi-GPU')
    parser.add_argument('--dataset', type=str, default='vqav2', 
                        choices=[
                            'vqav2', 'vizwiz', 'gqa', 'textvqa', 'ocrvqa',
                            'okvqa', 'aokvqa', 'sqa', 
                            'mme', 'mmbench', 'seedbench', 
                            'haloquest', 'mmhalbench'
                        ], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='configs/data.json')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to process')

    args = parser.parse_args()
    if args.dataset in ['vqav2', 'vizwiz', 'textvqa', 'sqa', 'okvqa', 'aokvqa', 'vcr']:
        args.split = 'val'
    elif args.dataset in ['gqa']:
        args.split = 'testdev'
    elif args.dataset in ['mme', 'mmbench', 'seedbench', 'haloquest', 'mmhalbench']:
        args.split = None
    elif args.dataset in ['ocrvqa', 'vsr']:
        args.split = 'test'   
    main(args)