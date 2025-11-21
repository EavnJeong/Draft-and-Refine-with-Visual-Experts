import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple


@torch.no_grad()
def compute_uq_caption_fidelity(
    drafts: List[str],
    masks: torch.Tensor,
    grid_hw: Tuple[int, int],
    vlm,
    vlm_processor,
    infer_vlm,
    dist_model,  # SBERT model
    batch,
    args,
    conf_alpha: float = 1.0,
    alpha: float = None,
):
    """
    Captioning Uq (fidelity version) for SBERT
    - Measures how consistent draft captions remain under masked images.
    - High fidelity → model relies on globally consistent visual evidence.
    """

    device = next(vlm.parameters()).device
    masks = masks.to(device)
    H, W = grid_hw

    # Encode draft captions with SBERT
    orig_embs = dist_model.encode(drafts, convert_to_tensor=True, device=device)
    orig_embs = orig_embs / orig_embs.norm(dim=-1, keepdim=True)

    # Precompute VLM inputs
    inputs_list = infer_vlm(vlm, vlm_processor, batch=batch, return_inputs=True)
    per_mask_fid, per_mask_conf = [], []

    for mask_2d in masks:
        masked_images = []
        for img_path, m in zip(batch["images"], mask_2d):
            img = img_path if isinstance(img_path, Image.Image) else Image.open(img_path).convert("RGB")
            H_img, W_img = img.height, img.width
            m_resized = F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), size=(H_img, W_img), mode="nearest")[0, 0].cpu().numpy()
            masked_np = (np.array(img) * m_resized[..., None]).astype("uint8")
            masked_images.append(Image.fromarray(masked_np))

        masked_batch = {"images": masked_images, **{k: v for k, v in batch.items() if k != "images"}}
        masked_drafts = infer_vlm(vlm, vlm_processor, masked_batch)

        # Encode masked captions with SBERT
        masked_embs = dist_model.encode(masked_drafts, convert_to_tensor=True, device=device)
        masked_embs = masked_embs / masked_embs.norm(dim=-1, keepdim=True)

        # Cosine similarity between original and masked captions
        cos_sim = F.cosine_similarity(orig_embs, masked_embs, dim=-1)
        fidelity = cos_sim.clamp(0, 1)
        conf = cos_sim.clamp_min(0.0) ** conf_alpha

        per_mask_fid.append(fidelity)
        per_mask_conf.append(conf)

    per_mask_fid = torch.stack(per_mask_fid, dim=0)
    per_mask_conf = torch.stack(per_mask_conf, dim=0)
    conf_norm = per_mask_conf / (per_mask_conf.sum(dim=0, keepdim=True) + 1e-8)

    uq_scores = (conf_norm * per_mask_fid).sum(dim=0)
    return uq_scores


@torch.no_grad()
def compute_uq_caption_faithfulness(
    drafts: List[str],
    masks: torch.Tensor,
    grid_hw: Tuple[int, int],
    vlm,
    vlm_processor,
    infer_vlm,
    dist_model,  # SBERT model
    batch,
    args,
    conf_alpha: float = 1.0,
    alpha: float = None,
):
    """
    Captioning Uq (faithfulness version) for SBERT
    - Measures how much the draft caption meaning changes when regions are masked.
    - Low faithfulness → draft heavily depends on specific region.
    """

    device = next(vlm.parameters()).device
    masks = masks.to(device)
    H, W = grid_hw

    # Encode original drafts with SBERT
    orig_embs = dist_model.encode(drafts, convert_to_tensor=True, device=device)
    orig_embs = orig_embs / orig_embs.norm(dim=-1, keepdim=True)

    inputs_list = infer_vlm(vlm, vlm_processor, batch=batch, return_inputs=True)
    per_mask_dists, per_mask_conf = [], []

    for mask_2d in masks:
        masked_images = []
        for img_path, m in zip(batch["images"], mask_2d):
            img = img_path if isinstance(img_path, Image.Image) else Image.open(img_path).convert("RGB")
            H_img, W_img = img.height, img.width
            m_resized = F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), size=(H_img, W_img), mode="nearest")[0, 0].cpu().numpy()
            masked_np = (np.array(img) * m_resized[..., None]).astype("uint8")
            masked_images.append(Image.fromarray(masked_np))

        masked_batch = {"images": masked_images, **{k: v for k, v in batch.items() if k != "images"}}
        masked_drafts = infer_vlm(vlm, vlm_processor, masked_batch)

        masked_embs = dist_model.encode(masked_drafts, convert_to_tensor=True, device=device)
        masked_embs = masked_embs / masked_embs.norm(dim=-1, keepdim=True)

        cos_sim = F.cosine_similarity(orig_embs, masked_embs, dim=-1)
        dist = 1.0 - cos_sim
        conf = cos_sim.clamp_min(0.0) ** conf_alpha

        per_mask_dists.append(dist)
        per_mask_conf.append(conf)

    per_mask_dists = torch.stack(per_mask_dists, dim=0)
    per_mask_conf = torch.stack(per_mask_conf, dim=0)
    conf_norm = per_mask_conf / (per_mask_conf.sum(dim=0, keepdim=True) + 1e-8)
    uq_scores = (conf_norm * per_mask_dists).sum(dim=0)

    return uq_scores


@torch.no_grad()
def compute_uq_caption_hybrid(
    drafts: List[str],
    masks: torch.Tensor,
    grid_hw: Tuple[int, int],
    vlm,
    vlm_processor,
    infer_vlm,
    dist_model,         # SBERT model
    batch,
    args,
    conf_alpha: float = 1.0,
    alpha: float = None,     # adaptive ratio
):
    """
    Hybrid UQ computation for captioning (DnR draft-based)
    Combines fidelity (consistency) and faithfulness (dependency)
    using SBERT for semantic similarity.
    """

    num_masks = masks.shape[0] // 2
    topk_masks, bottomk_masks = masks[:num_masks], masks[num_masks:]

    # 1️⃣ Compute UQ (fidelity / faithfulness)
    uq_top = compute_uq_caption_fidelity(
        drafts, topk_masks, grid_hw, vlm, vlm_processor, infer_vlm,
        dist_model, batch, args, conf_alpha
    )
    uq_bottom = compute_uq_caption_faithfulness(
        drafts, bottomk_masks, grid_hw, vlm, vlm_processor, infer_vlm,
        dist_model, batch, args, conf_alpha
    )

    # 2️⃣ Load relevance maps (object-centric)
    r_map = batch["r_map"]
    r_map_tensor = torch.stack([torch.load(f) for f in r_map])
    min_vals = r_map_tensor.amin(dim=(-2, -1), keepdim=True)
    max_vals = r_map_tensor.amax(dim=(-2, -1), keepdim=True)
    r_map_tensor = (r_map_tensor - min_vals) / (max_vals - min_vals + 1e-8)
    r_map_tensor = r_map_tensor.to(uq_top.device).float()

    # 3️⃣ Compute or use provided α
    if alpha is None:
        alpha = compute_adaptive_alpha_captioning(
            r_map_tensor,
            drafts=drafts,
            dist_model=dist_model
        ).to(uq_top.device)  # [B]

    # 4️⃣ Combine weighted UQ components
    uq_hybrid = alpha * uq_top + (1 - alpha) * uq_bottom
    return uq_hybrid


@torch.no_grad()
def compute_adaptive_alpha_captioning(
    r_map_tensor: torch.Tensor,
    drafts=None,
    dist_model=None,          # SBERT model
    beta_entropy: float = 0.5,
    beta_contrast: float = 0.3,
    beta_semantic: float = 0.2,
):
    """
    Adaptive α computation for Captioning (SBERT version)
    - entropy: measures focus/sharpness of relevance
    - contrast: measures intensity difference between salient and non-salient regions
    - semantic: measures how spatially balanced the caption meaning is w.r.t. visual center

    Args:
        r_map_tensor (Tensor): (B, H, W) object-centric relevance maps.
        drafts (list[str]): draft captions.
        dist_model (SentenceTransformer): SBERT model for encoding text.
        beta_* (float): weights for entropy, contrast, semantic factors.

    Returns:
        alpha (Tensor): (B,) adaptive weighting ∈ [0,1]
    """
    B, H, W = r_map_tensor.shape
    device = r_map_tensor.device

    # 1️⃣ Normalize relevance map
    r_norm = r_map_tensor.clamp_min(0)
    r_norm = r_norm / (r_norm.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    # 2️⃣ Entropy (map sharpness)
    r_flat = r_norm.view(B, -1)
    entropy = -(r_flat * (r_flat + 1e-8).log()).sum(dim=1)
    max_entropy = torch.log(torch.tensor(r_flat.shape[1], device=device))
    H_norm = 1 - (entropy / max_entropy).clamp(0, 1)  # sharper map → higher α

    # 3️⃣ Contrast (top vs bottom intensity)
    k = max(1, int(0.25 * H * W))
    sorted_vals, _ = torch.sort(r_flat, dim=1, descending=True)
    mean_top = sorted_vals[:, :k].mean(dim=1)
    mean_bottom = sorted_vals[:, -k:].mean(dim=1)
    contrast = ((mean_top - mean_bottom) / (mean_top + mean_bottom + 1e-8)).clamp(0, 1)

    # 4️⃣ Semantic alignment (draft caption meaning vs relevance balance)
    if drafts is not None and dist_model is not None:
        text_emb = dist_model.encode(drafts, convert_to_tensor=True, device=device)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        ), dim=-1)  # (H, W, 2)
        coords = coords.unsqueeze(0).expand(B, H, W, 2)
        weights = r_norm.unsqueeze(-1)
        center = (coords * weights).sum(dim=(1, 2))  # (B, 2)

        # Larger norm → attention biased to one side → high semantic component
        semantic = center.norm(dim=-1).clamp(0, 1)
    else:
        semantic = torch.zeros_like(H_norm)

    # 5️⃣ Combine all factors
    alpha = (
        beta_entropy * H_norm +
        beta_contrast * contrast +
        beta_semantic * semantic
    )
    alpha = alpha / (beta_entropy + beta_contrast + beta_semantic)
    alpha = alpha.clamp(0, 1)

    return alpha