def update_args(args):
    if isinstance(args.vlm_device, int) and args.vlm_device >= 0:
        args.vlm_device = f"cuda:{args.vlm_device}"

    if args.expert_device >= 0:
        args.expert_device = f"cuda:{args.expert_device}"
    
    if args.dataset in ['vqav2', 'vizwiz', 'textvqa', 'okvqa', 'aokvqa', 'sqa', 'vcr']:
        args.split = 'val'
    elif args.dataset in ['gqa', 'ocrvqa']:
        args.split = 'testdev'
    elif args.dataset in ['vsr']:
        args.split = 'test'
    elif args.dataset in ['cococaption', 'nocaps', 'flickr']:
        args.split = 'val'
    elif args.dataset in ['mme', 'mmbench', 'seedbench', 'haloquest', 'mmhalbench']:
        args.split = None
    
    if args.dataset in ['vqav2', 'vizwiz', 'gqa', 'textvqa', 'ocrvqa', 'vcr', 'vsr', 'okvqa', 'aokvqa', 'sqa', 'mme', 'mmbench', 'seedbench', 'haloquest', 'mmhalbench']:
        args.task = 'vqa'
    elif args.dataset in ['cococaption', 'nocaps', 'flickr']:
        args.task = 'caption'