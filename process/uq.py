import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from typing import List, Tuple 


@torch.no_grad()
def compute_uq_relevance_fidelity(
    orig_ans: List[str],
    masks: torch.Tensor,
    grid_hw: Tuple[int, int],
    vlm,
    vlm_processor,
    infer_vlm,
    dist_model,
    dist_tokenizer,
    batch,
    args,
    conf_alpha: float = 1.0,
    alpha: float = None
):
    device = next(vlm.parameters()).device
    masks = masks.to(device)
    H, W = grid_hw

    if getattr(args, "dataset", None) == "sqa":
        choice_text_map = {"A": "option A", "B": "option B", "C": "option C", "D": "option D", "E": "option E"}
        orig_ans_texts = [choice_text_map.get(a.strip().upper(), a) for a in orig_ans]
    else:
        orig_ans_texts = orig_ans
    tokens = dist_tokenizer(orig_ans_texts).to(device)
    orig_embs = dist_model.encode_text(tokens)
    orig_embs = orig_embs / orig_embs.norm(dim=-1, keepdim=True)

    # Encode original answer with CLIP
    # tokens = dist_tokenizer(orig_ans).to(device)
    # orig_embs = dist_model.encode_text(tokens)
    # orig_embs = orig_embs / orig_embs.norm(dim=-1, keepdim=True)

    # Prepare original image tensor
    inputs_list = infer_vlm(vlm, vlm_processor, batch=batch, return_inputs=True)
    per_mask_fid, per_mask_conf = [], []

    for mask_2d in masks:
        # =======================
        # Model-specific masking
        # =======================
        if args.vlm == "qwen_vl":
            masked_images = []
            for img_path, m in zip(batch['images'], mask_2d):
                img = img_path if isinstance(img_path, Image.Image) else Image.open(img_path).convert("RGB")
                H_img, W_img = img.height, img.width
                m_resized = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(), size=(H_img, W_img), mode="nearest"
                )[0, 0].cpu().numpy()
                masked_np = (np.array(img) * m_resized[..., None]).astype("uint8")
                masked_images.append(Image.fromarray(masked_np))
            masked_batch = {"images": masked_images, **{k: v for k, v in batch.items() if k != "images"}}
            if "choices" in batch:
                masked_batch["choices"] = batch["choices"]
            if "lectures" in batch:
                masked_batch["lectures"] = batch["lectures"]
            if "contexts" in batch:
                masked_batch["contexts"] = batch["contexts"]

        elif args.vlm == "cogvlm":
            masked_images = []
            for img_path, m in zip(batch['images'], mask_2d):
                img = img_path if isinstance(img_path, Image.Image) else Image.open(img_path).convert("RGB")
                H_img, W_img = img.height, img.width

                m_resized = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(), size=(H_img, W_img), mode="nearest"
                )[0, 0].cpu().numpy()

                img_np = np.array(img)
                masked_np = (img_np * m_resized[..., None]).astype("uint8")

                masked_images.append(Image.fromarray(masked_np))

            masked_batch = {
                "images": masked_images,
                **{k: v for k, v in batch.items() if k != "images"}
            }
            if "choices" in batch:
                masked_batch["choices"] = batch["choices"]
            if "lectures" in batch:
                masked_batch["lectures"] = batch["lectures"]
            if "contexts" in batch:
                masked_batch["contexts"] = batch["contexts"]

        else:
            if args.vlm == 'llava':
                img_tensor = torch.cat([inp["pixel_values"][:, 0] for inp in inputs_list], dim=0).to(device)
            else:
                img_tensor = torch.cat([inp["pixel_values"] for inp in inputs_list], dim=0).to(device)
            H_img, W_img = img_tensor.shape[-2:]

            pix_mask = (
                F.interpolate(mask_2d.unsqueeze(1).float(), size=grid_hw, mode="nearest").to(img_tensor.dtype)
            )

            if args.vlm in ['llava', 'instructblip', 'pali', 'minigptv2']:
                masked_img = img_tensor * pix_mask
            elif args.vlm == 'idefics':
                masked_img = img_tensor.squeeze() * pix_mask
            else:
                masked_img = img_tensor * pix_mask

            masked_batch = {
                'images': [Image.fromarray((m.permute(1, 2, 0).cpu().numpy()*255).astype('uint8')) for m in masked_img],
                **{k: v for k, v in batch.items() if k != "images"}
            }
            if "choices" in batch:
                masked_batch["choices"] = batch["choices"]
            if "lectures" in batch:
                masked_batch["lectures"] = batch["lectures"]
            if "contexts" in batch:
                masked_batch["contexts"] = batch["contexts"]
        # =======================
        # Inference for masked input
        # =======================
        masked_ans = infer_vlm(vlm, vlm_processor, masked_batch)

        # Encode masked answers with CLIP
        tokens_masked = dist_tokenizer(masked_ans).to(device)
        masked_embs = dist_model.encode_text(tokens_masked)
        masked_embs = masked_embs / masked_embs.norm(dim=-1, keepdim=True)

        # =======================
        # Fidelity 
        # =======================
        cos_sim = F.cosine_similarity(orig_embs, masked_embs, dim=-1)
        fidelity = cos_sim.clamp(0, 1)
        conf = cos_sim.clamp_min(0.0) ** conf_alpha

        per_mask_fid.append(fidelity)
        per_mask_conf.append(conf)

    # Stack results
    per_mask_fid = torch.stack(per_mask_fid, dim=0)
    per_mask_conf = torch.stack(per_mask_conf, dim=0)
    conf_norm = per_mask_conf / (per_mask_conf.sum(dim=0, keepdim=True) + 1e-8)

    uq_scores = (conf_norm * per_mask_fid).sum(dim=0)
    return uq_scores


@torch.no_grad()
def compute_uq_relevance_faithfulness(
    orig_ans: List[str],
    masks: torch.Tensor,
    grid_hw: Tuple[int, int],
    vlm,
    vlm_processor,
    infer_vlm,
    dist_model,
    dist_tokenizer,
    batch,
    args,
    conf_alpha: float = 1.0,  # confidence scaling
    alpha: float = None
):
    device = next(vlm.parameters()).device
    masks = masks.to(device)
    H, W = grid_hw

    if getattr(args, "dataset", None) == "sqa":
        choice_text_map = {"A": "option A", "B": "option B", "C": "option C", "D": "option D", "E": "option E"}
        orig_ans_texts = [choice_text_map.get(a.strip().upper(), a) for a in orig_ans]
    else:
        orig_ans_texts = orig_ans
    tokens = dist_tokenizer(orig_ans_texts).to(device)
    orig_embs = dist_model.encode_text(tokens)
    orig_embs = orig_embs / orig_embs.norm(dim=-1, keepdim=True)

    # 1️⃣ Encode original answer with distance model text encoder
    # tokens = dist_tokenizer(orig_ans).to(device)
    # orig_embs = dist_model.encode_text(tokens)
    # orig_embs = orig_embs / orig_embs.norm(dim=-1, keepdim=True)

    # 2️⃣ Prepare image tensor
    inputs_list = infer_vlm(vlm, vlm_processor, batch=batch, return_inputs=True)

    per_mask_dists = []
    per_mask_conf = []

    # 3️⃣ Iterate over masks
    for mask_2d in masks:
        # --- Model-specific masking logic ---
        if args.vlm == "qwen_vl":
            masked_images = []
            for img_path, m in zip(batch['images'], mask_2d):
                if isinstance(img_path, Image.Image):
                    img = img_path
                else:
                    img = Image.open(img_path).convert("RGB")
                H_img, W_img = img.height, img.width

                m_resized = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=(H_img, W_img),
                    mode="nearest"
                )[0, 0].cpu().numpy()

                img_np = np.array(img)
                masked_np = (img_np * m_resized[..., None]).astype("uint8")
                masked_images.append(Image.fromarray(masked_np))

            masked_batch = {
                "images": masked_images,
                **{k: v for k, v in batch.items() if k != "images"}
            }
            if "choices" in batch:
                masked_batch["choices"] = batch["choices"]
            if "lectures" in batch:
                masked_batch["lectures"] = batch["lectures"]
            if "contexts" in batch:
                masked_batch["contexts"] = batch["contexts"]

        elif args.vlm == "cogvlm":
            masked_images = []
            for img_path, m in zip(batch['images'], mask_2d):
                img = img_path if isinstance(img_path, Image.Image) else Image.open(img_path).convert("RGB")
                H_img, W_img = img.height, img.width

                m_resized = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(), size=(H_img, W_img), mode="nearest"
                )[0, 0].cpu().numpy()

                img_np = np.array(img)
                masked_np = (img_np * m_resized[..., None]).astype("uint8")

                masked_images.append(Image.fromarray(masked_np))

            masked_batch = {
                "images": masked_images,
                **{k: v for k, v in batch.items() if k != "images"}
            }
            if "choices" in batch:
                masked_batch["choices"] = batch["choices"]
            if "lectures" in batch:
                masked_batch["lectures"] = batch["lectures"]
            if "contexts" in batch:
                masked_batch["contexts"] = batch["contexts"]

        else:
            if args.vlm == 'llava':
                img_tensor = torch.cat([inp["pixel_values"][:, 0] for inp in inputs_list], dim=0).to(device)
            else:
                img_tensor = torch.cat([inp["pixel_values"] for inp in inputs_list], dim=0).to(device)
            H_img, W_img = img_tensor.shape[-2:]

            pix_mask = (
                F.interpolate(
                    mask_2d.unsqueeze(1).float(),
                    size=grid_hw,
                    mode="nearest"
                ).to(img_tensor.dtype)
            )

            if args.vlm in ['llava', 'instructblip', 'pali', 'minigptv2']:
                masked_img = img_tensor * pix_mask
            elif args.vlm == 'idefics':
                masked_img = img_tensor.squeeze() * pix_mask
            else:
                masked_img = img_tensor * pix_mask

            masked_batch = {
                'images': [
                    Image.fromarray((m.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                    for m in masked_img
                ],
                **{k: v for k, v in batch.items() if k != "images"}
            }
            if "choices" in batch:
                masked_batch["choices"] = batch["choices"]
            if "lectures" in batch:
                masked_batch["lectures"] = batch["lectures"]
            if "contexts" in batch:
                masked_batch["contexts"] = batch["contexts"]

        # --- Inference for masked input ---
        masked_ans = infer_vlm(vlm, vlm_processor, masked_batch)

        # 4️⃣ Encode masked answers using distance model
        tokens_masked = dist_tokenizer(masked_ans).to(device)
        masked_embs = dist_model.encode_text(tokens_masked)
        masked_embs = masked_embs / masked_embs.norm(dim=-1, keepdim=True)

        # 5️⃣ Compute cosine similarity & confidence weighting
        cos_sim = F.cosine_similarity(orig_embs, masked_embs, dim=-1)
        dist = 1.0 - cos_sim                          # semantic shift
        conf = cos_sim.clamp_min(0.0) ** conf_alpha   # confidence weight

        per_mask_dists.append(dist)
        per_mask_conf.append(conf)

    # 6️⃣ Stack & confidence-weighted aggregation
    per_mask_dists = torch.stack(per_mask_dists, dim=0)
    per_mask_conf = torch.stack(per_mask_conf, dim=0)

    # Normalize confidence per sample
    conf_norm = per_mask_conf / (per_mask_conf.sum(dim=0, keepdim=True) + 1e-8)

    # Weighted mean
    uq_scores = (conf_norm * per_mask_dists).sum(dim=0)

    return uq_scores


@torch.no_grad()
def compute_uq_hybrid(
    orig_ans: List[str],
    masks: torch.Tensor,
    grid_hw: Tuple[int, int],
    vlm,
    vlm_processor,
    infer_vlm,
    dist_model,
    dist_tokenizer,
    batch,
    args,
    conf_alpha: float = 1.0,
    alpha = 0.5,               # blending ratio
):
    num_masks = masks.shape[0] // 2
    topk_masks, bottomk_masks = masks[:num_masks], masks[num_masks:]

    uq_top = compute_uq_relevance_fidelity(
        orig_ans, topk_masks, grid_hw, vlm, vlm_processor, infer_vlm, 
        dist_model, dist_tokenizer,
        batch, args, conf_alpha
    )
    uq_bottom = compute_uq_relevance_fidelity(
        orig_ans, bottomk_masks, grid_hw, vlm, vlm_processor, infer_vlm, 
        dist_model, dist_tokenizer,
        batch, args, conf_alpha
    )

    r_map = batch['r_map']
    r_map_tensor = torch.stack([torch.load(f) for f in r_map])
    min_vals = r_map_tensor.amin(dim=(-2, -1), keepdim=True)
    max_vals = r_map_tensor.amax(dim=(-2, -1), keepdim=True)
    r_map_tensor = (r_map_tensor - min_vals) / (max_vals - min_vals + 1e-8)
    r_map_tensor = r_map_tensor.to(uq_top.device).float()

    if alpha is None:
        alpha = compute_adaptive_alpha(
            r_map_tensor,
            q_texts=batch["questions"],
            dist_model=dist_model,
            dist_tokenizer=dist_tokenizer
        ).to(uq_top.device)  # [B]

    uq_hybrid = alpha * uq_top + (1 - alpha) * uq_bottom
    return uq_hybrid


@torch.no_grad()
def compute_adaptive_alpha(
        r_map_tensor, 
        q_texts=None, 
        dist_model=None, 
        dist_tokenizer=None, 
        beta_entropy=0.5, 
        beta_contrast=0.3, 
        beta_semantic=0.2
    ):
    """
    Improved Adaptive α Computation
    - entropy: measures sharpness or focus of attention
    - contrast: captures intensity difference between top and bottom regions
    - semantic: quantifies how well the relevance map aligns semantically with the question

    Args:
        r_map_tensor: (B, H, W)
            Relevance maps for each sample.
        q_texts: (optional) list of question strings, used for semantic alignment.
        clip_model, tokenizer: (optional) CLIP model and tokenizer for semantic similarity computation.
        beta_*: weighting coefficients for each feature component.

    Returns:
        alpha: (B,) ∈ [0,1]
            Adaptive weighting factor per sample.
    """
    B, H, W = r_map_tensor.shape
    device = r_map_tensor.device

    # 1️⃣ Normalize
    r_norm = r_map_tensor.clamp_min(0)
    r_norm = r_norm / (r_norm.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    # 2️⃣ Entropy (sharpness)
    r_flat = r_norm.view(B, -1)
    entropy = -(r_flat * (r_flat + 1e-8).log()).sum(dim=1)
    max_entropy = torch.log(torch.tensor(r_flat.shape[1], device=device))
    H_norm = 1 - (entropy / max_entropy).clamp(0, 1)  # sharp → high α

    # 3️⃣ Contrast (top vs bottom intensity gap)
    k = max(1, int(0.25 * H * W))
    sorted_vals, _ = torch.sort(r_flat, dim=1, descending=True)
    mean_top = sorted_vals[:, :k].mean(dim=1)
    mean_bottom = sorted_vals[:, -k:].mean(dim=1)
    contrast = ((mean_top - mean_bottom) / (mean_top + mean_bottom + 1e-8)).clamp(0, 1)

    # 4️⃣ Semantic alignment (optional, if distance model inputs given)
    if q_texts is not None and dist_model is not None and dist_tokenizer is not None:
        q_tokens = dist_tokenizer(q_texts).to(device)
        q_emb = dist_model.encode_text(q_tokens)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        ), dim=-1)  # (H, W, 2)
        coords = coords.unsqueeze(0).expand(B, H, W, 2)
        weights = r_norm.unsqueeze(-1)
        center = (coords * weights).sum(dim=(1, 2))  # (B, 2)

        semantic = center.norm(dim=-1).clamp(0, 1)
    else:
        semantic = torch.zeros_like(H_norm)

    # 5️⃣ Combine features
    alpha = (
        beta_entropy * H_norm +
        beta_contrast * contrast +
        beta_semantic * semantic
    )
    alpha = alpha / (beta_entropy + beta_contrast + beta_semantic)
    alpha = alpha.clamp(0, 1)

    return alpha
