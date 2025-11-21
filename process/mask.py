import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_gumbel_topk_masks(
    r_map, grid_hw,
    num_masks=8,
    area_ratio=0.25,
    beta=2.0,
    device='cpu'
):
    H, W = grid_hw
    num_patches = H * W
    k = max(1, min(num_patches, int(area_ratio * num_patches)))

    # Load relevance map from file paths
    r_map_tensor = torch.stack([torch.load(f) for f in r_map])
    r_vec = r_map_tensor.to(device).reshape(r_map_tensor.shape[0], -1).float()

    # Sharpen and normalize the distribution
    r_pow = (r_vec.clamp_min(1e-12))**beta
    r_pow = r_pow / (r_pow.sum(dim=1, keepdim=True) + 1e-12)

    B = r_pow.shape[0]
    all_masks = []
    for _ in range(num_masks):
        # Gumbel-Top-k sampling to select indices to mask
        gumbel_noise = -torch.empty_like(r_pow).exponential_().log()
        scores = r_pow.log().add(gumbel_noise)
        topk_idx = scores.topk(k, largest=True).indices

        # Create a mask where top-k indices are set to 0 (masked out)
        mask_1d = torch.ones_like(r_pow)
        mask_1d.scatter_(1, topk_idx, 0.0)
        mask_2d = mask_1d.view(B, H, W)
        all_masks.append(mask_2d)

    # Stack all generated masks into a single tensor
    return torch.stack(all_masks, dim=0)


@torch.no_grad()
def generate_gumbel_bottomk_masks(
        r_map, grid_hw,
        num_masks=8,
        area_ratio=0.25,
        beta=2.0,
        device='cpu'
    ):
    H, W = grid_hw
    num_patches = H * W
    k = max(1, min(num_patches, int(area_ratio * num_patches)))

    # Load relevance maps
    r_map_tensor = torch.stack([torch.load(f) for f in r_map])
    r_vec = r_map_tensor.to(device).reshape(r_map_tensor.shape[0], -1).float()

    # Normalize to probability distribution
    r_pow = (r_vec.clamp_min(1e-12)) ** beta
    r_pow = r_pow / (r_pow.sum(dim=1, keepdim=True) + 1e-12)

    B = r_pow.shape[0]
    all_masks = []

    for _ in range(num_masks):
        gumbel_noise = -torch.empty_like(r_pow).exponential_().log()
        scores = r_pow.log().add(gumbel_noise)

        bottomk_idx = scores.topk(k, largest=False).indices

        mask_1d = torch.ones_like(r_pow)
        mask_1d.scatter_(1, bottomk_idx, 0.0)
        mask_2d = mask_1d.view(B, H, W)
        all_masks.append(mask_2d)

    return torch.stack(all_masks, dim=0)  # [num_masks, B, H, W]


@torch.no_grad()
def generate_gumbel_hybrid_masks(
    r_map,
    grid_hw,
    num_masks=8,
    area_ratio=0.25,
    beta=2.0,
    device='cpu'
):
    topk_masks = generate_gumbel_topk_masks(
        r_map, grid_hw, num_masks, area_ratio, beta, device
    )
    bottomk_masks = generate_gumbel_bottomk_masks(
        r_map, grid_hw, num_masks, area_ratio, beta, device
    )

    hybrid_masks = torch.cat([topk_masks, bottomk_masks], dim=0)
    return hybrid_masks


@torch.no_grad()
def generate_dense_topk_masks(
    r_map, grid_hw,
    patch_size=32,
    num_masks=8,
    area_ratio=0.25,
    beta=2.0,
    device='cpu'
):
    H, W = grid_hw
    num_patches_H = H // patch_size
    num_patches_W = W // patch_size
    num_regions = num_patches_H * num_patches_W
    k = max(1, min(num_regions, int(area_ratio * num_regions)))

    # Load and flatten relevance maps
    r_map_tensor = torch.stack([torch.load(f) for f in r_map])
    min_vals = r_map_tensor.amin(dim=(-2, -1), keepdim=True)
    max_vals = r_map_tensor.amax(dim=(-2, -1), keepdim=True)
    r_map_tensor = (r_map_tensor - min_vals) / (max_vals - min_vals + 1e-8)

    B = r_map_tensor.shape[0]

    region_relevance = F.adaptive_avg_pool2d(r_map_tensor, (num_patches_H, num_patches_W))
    region_vec = region_relevance.view(B, -1)

    r_pow = (region_vec.clamp_min(1e-12)) ** beta
    r_pow = r_pow / (r_pow.sum(dim=1, keepdim=True) + 1e-12)

    all_masks = []
    for _ in range(num_masks):
        gumbel_noise = -torch.empty_like(r_pow).exponential_().log()
        scores = r_pow.log() + gumbel_noise

        topk_idx = scores.topk(k, largest=True).indices

        region_mask = torch.ones_like(r_pow)
        region_mask.scatter_(1, topk_idx, 0.0)
        region_mask_2d = region_mask.view(B, num_patches_H, num_patches_W)

        mask_2d = F.interpolate(region_mask_2d.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
        all_masks.append(mask_2d)

    all_masks = torch.stack(all_masks, dim=0)
    return all_masks  # [num_masks, B, H, W]


@torch.no_grad()
def generate_dense_bottomk_masks(
    r_map, grid_hw,
    patch_size=16,
    num_masks=8,
    area_ratio=0.25,
    beta=2.0,
    device='cpu'
):
    H, W = grid_hw
    num_patches_H = H // patch_size
    num_patches_W = W // patch_size
    num_regions = num_patches_H * num_patches_W
    k = max(1, min(num_regions, int(area_ratio * num_regions)))

    # Load relevance maps
    r_map_tensor = torch.stack([torch.load(f) for f in r_map])
    min_vals = r_map_tensor.amin(dim=(-2, -1), keepdim=True)
    max_vals = r_map_tensor.amax(dim=(-2, -1), keepdim=True)
    r_map_tensor = (r_map_tensor - min_vals) / (max_vals - min_vals + 1e-8)

    B = r_map_tensor.shape[0]

    region_relevance = F.adaptive_avg_pool2d(r_map_tensor, (num_patches_H, num_patches_W))
    region_vec = region_relevance.view(B, -1)

    # Sharpen & Normalize
    r_pow = (region_vec.clamp_min(1e-12)) ** beta
    r_pow = r_pow / (r_pow.sum(dim=1, keepdim=True) + 1e-12)

    all_masks = []
    for _ in range(num_masks):
        gumbel_noise = -torch.empty_like(r_pow).exponential_().log()
        scores = r_pow.log() + gumbel_noise

        bottomk_idx = scores.topk(k, largest=False).indices

        region_mask = torch.ones_like(r_pow)
        region_mask.scatter_(1, bottomk_idx, 0.0)
        region_mask_2d = region_mask.view(B, num_patches_H, num_patches_W)

        mask_2d = F.interpolate(region_mask_2d.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
        all_masks.append(mask_2d)

    # Stack all masks
    all_masks = torch.stack(all_masks, dim=0)
    return all_masks  # [num_masks, B, H, W]
