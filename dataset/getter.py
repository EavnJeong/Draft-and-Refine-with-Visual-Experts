import os
from torch.utils.data import DataLoader


def get_qa_loader(
    args, 
    data_path, 
    split='train', 
    batch_size=32, 
    num_workers=4
):
    if args.dataset == 'vqav2':
        from .collate_fns import vqav2_collate_fn
        from .vqav2 import VQAv2Dataset
        dataset = VQAv2Dataset(
            root_dir=data_path,
            split=split,
            limit=args.limit,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=vqav2_collate_fn,
        )
        return dataloader

    elif args.dataset == 'gqa':
        from .collate_fns import gqa_collate_fn
        from .gqa import GQADataset
        dataset = GQADataset(
            root_dir=data_path,
            split=split,
            balanced=True,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=gqa_collate_fn,
        )
        return dataloader
    
    elif args.dataset == 'vizwiz':
        from .vizwiz import VizwizDataset
        from .collate_fns import vizwiz_collate_fn
        
        root_dir = os.path.join(data_path, 'images', split)
        ann_file = os.path.join(data_path, 'annotations', f'{split}.json')

        dataset = VizwizDataset(
            root_dir=root_dir,
            ann_file=ann_file,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=vizwiz_collate_fn,
        )
        return dataloader

    elif args.dataset == 'textvqa':
        from .collate_fns import textvqa_collate_fn
        from .textvqa import TextVQADataset
        dataset = TextVQADataset(
            root_dir=data_path,
            split=split,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=textvqa_collate_fn,
        )
        return dataloader

    elif args.dataset == 'ocrvqa':
        from .collate_fns import ocrvqa_collate_fn
        from .ocrvqa import OCRVQADataset
        dataset = OCRVQADataset(
            root_dir=data_path,
            split=split,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=ocrvqa_collate_fn,
        )
        return dataloader

    elif args.dataset == 'vcr':
        from .collate_fns import vcr_collate_fn
        from .vcr import VCRDataset
        dataset = VCRDataset(
            root_dir=data_path,
            split=split,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=vcr_collate_fn,
        )
        return dataloader

    elif args.dataset == 'vsr':
        from .collate_fns import vsr_collate_fn
        from .vsr import VSRDataset
        dataset = VSRDataset(
            root_dir=data_path,
            split=split,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=vsr_collate_fn,
        )
        return dataloader

    elif args.dataset == 'okvqa':
        from .collate_fns import okvqa_collate_fn
        from .okvqa import OKVQADataset
        dataset = OKVQADataset(
            root_dir=data_path,
            split=split,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=okvqa_collate_fn,
        )
        return dataloader
    elif args.dataset == 'aokvqa':
        from .collate_fns import aokvqa_collate_fn
        from .aokvqa import AOKVQADataset
        dataset = AOKVQADataset(
            root_dir=data_path,
            split=split,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=aokvqa_collate_fn,
        )
        return dataloader

    elif args.dataset == 'sqa':
        from .collate_fns import sqa_collate_fn
        from .sqa import ScienceQADataset
        dataset = ScienceQADataset(
            root_dir=data_path,
            split=split,
            options=["A", "B", "C", "D", "E"],
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=sqa_collate_fn,
        )
        return dataloader

    elif args.dataset == "mme":
        from .collate_fns import mme_collate_fn
        from .mme import MMEAllDataset
        dataset = MMEAllDataset(root_dir=data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=mme_collate_fn,
        )
        return dataloader
    elif args.dataset == 'mmbench':
        from .collate_fns import mmbench_collate_fn
        from .mmbench import MMBenchDataset
        dataset = MMBenchDataset(
            root_dir=data_path,
            limit_per_task=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=mmbench_collate_fn,
        )
        return dataloader
    elif args.dataset == 'seedbench':
        from .collate_fns import seedbench_collate_fn
        from .seedbench import SEEDBenchDataset
        dataset = SEEDBenchDataset(
            root_dir=data_path,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=seedbench_collate_fn,
        )
        return dataloader
        
    elif args.dataset == 'haloquest':
        from .collate_fns import haloquest_collate_fn
        from .haloquest import HaloQuestDataset
        dataset = HaloQuestDataset(
            root_dir=data_path,
            verify_images=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=haloquest_collate_fn,
        )
        return dataloader

    elif args.dataset == 'mmhalbench':
        from .collate_fns import collate_fn_mmhalbench
        from .mmhalbench import MMHalBenchDataset
        dataset = MMHalBenchDataset(
            root_dir=data_path,
            verify_images=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_mmhalbench,
        )
        return dataloader
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

        
def get_caption_loader(
    args,
    data_path,
    split='train',
    batch_size=32,
    num_workers=4,
):
    if args.dataset == 'cococaption':
        from .collate_fns import coco_collate_fn
        from .captioncoco import COCODatasetCaptioning
        dataset = COCODatasetCaptioning(
            root_dir=data_path,
            split=split,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=coco_collate_fn,
        )
        return dataloader
    elif args.dataset == 'nocaps':
        from .collate_fns import nocaps_collate_fn
        from .nocaps import NoCapsDataset
        dataset = NoCapsDataset(
            root_dir=data_path,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=nocaps_collate_fn,
        )
        return dataloader
    elif args.dataset == 'flickr':
        from .collate_fns import flickr_collate_fn
        from .flickr import FlickrDataset
        dataset = FlickrDataset(
            root_dir=data_path,
            split=split,
            limit=args.limit
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=flickr_collate_fn,
        )
        return dataloader
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")