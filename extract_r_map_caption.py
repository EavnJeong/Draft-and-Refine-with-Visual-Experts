import argparse
import json
from tqdm import tqdm

from dataset.getter import get_caption_loader
from process.rm import build_relevance_maps_captioning



def main(args):
    with open(args.data_path, 'r') as f:
        data_paths = json.load(f)['data_paths']

    val_loader = get_caption_loader(
        args,
        data_path=data_paths[args.dataset],
        split=args.split,
        batch_size=args.batch_size,
        num_workers=0
    )

    num = 0
    for batch in tqdm(val_loader):
        num = build_relevance_maps_captioning(
            args.dataset,
            batch['images'],
            device=args.device,
            num=num,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Agent CLI")
    parser.add_argument('--dataset', type=str, default='cococaption', choices=['cococaption', 'nocaps', 'flickr'], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='configs/data.json')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device ID to use')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to process')

    args = parser.parse_args()
    if args.dataset in ['cococaption', 'nocaps', 'flickr']:
        args.split = 'val'
    main(args)