from .loader import (
    load_grounding_dino, 
    load_sam,
    load_depth_anything_v2,
    load_mdetr,
)
from .predict import (
    run_grounding_dino,
    run_sam,
    run_depth_anything_v2_front,
    run_depth_anything_v2_back,
    run_mdetr,
)

expert_loader_map = {
    'grounding_dino': load_grounding_dino,
    'sam': load_sam,
    'depth_anything_v2_front': load_depth_anything_v2,
    'depth_anything_v2_back': load_depth_anything_v2,
    'mdetr': load_mdetr,
}

expert_run_map = {
    'grounding_dino': run_grounding_dino,
    'sam': run_sam,
    'depth_anything_v2_front': run_depth_anything_v2_front,
    'depth_anything_v2_back': run_depth_anything_v2_back,
    'mdetr': run_mdetr,
}


def get_expert(args, device):
    return expert_loader_map[args.expert](device)


def get_experts(args, device):
    experts = {}
    for name in args.experts:
        if name not in expert_loader_map:
            print(f"⚠️ Expert '{name}' is not supported.")
            continue
        print(f"🔧 Loading expert: {name}")
        expert = expert_loader_map[name](device)
        model, processor = expert['model'], expert['processor']
        experts[name] = {
            "model": model,
            "processor": processor,
            "run": expert_run_map[name]
        }

    return experts