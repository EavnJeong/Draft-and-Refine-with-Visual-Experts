from PIL import Image
import numpy as np
from PIL import Image
import cv2


def render_grounding_dino(
        image: Image.Image, 
        detections, 
        pad_ratio=0.1,
        mode="gray",
        blur_strength=25
    ) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)
    H, W = img.shape[:2]

    mask = np.zeros((H, W), dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        pad_x = int(pad_ratio * (x2 - x1))
        pad_y = int(pad_ratio * (y2 - y1))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x)
        y2 = min(H, y2 + pad_y)
        mask[y1:y2, x1:x2] = 1

    mask_3c = np.repeat(mask[..., None], 3, axis=-1)

    if mode == "black":
        out = np.where(mask_3c == 1, img, 0)
    elif mode == "blur":
        blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        out = np.where(mask_3c == 1, img, blurred)
    elif mode == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray3c = np.stack([gray]*3, axis=-1)
        out = np.where(mask_3c == 1, img, gray3c)
    elif mode == "white":
        white_bg = np.ones_like(img) * 255
        out = np.where(mask_3c == 1, img, white_bg)
    elif mode == "highlight":
        blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        highlight = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        out = np.where(mask_3c == 1, highlight, blurred)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return Image.fromarray(out)


def render_sam(
        image: Image.Image,
        masks: np.ndarray,   # [N, H, W] or [H, W]
        mode="gray",
        blur_strength=25,
        thr=0.5
    ) -> Image.Image:
    """
    Render image with SAM segmentation masks.
    - image: PIL Image
    - masks: np.ndarray of shape [N, H, W] or [H, W]
    - mode: "gray", "black", "white", "blur", "highlight"
    - blur_strength: kernel size for blurring
    - thr: threshold for mask binarization
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)
    H, W = img.shape[:2]

    if masks.ndim == 2:
        full_mask = (masks > thr).astype(np.uint8)
    elif masks.ndim == 3:
        full_mask = np.zeros((H, W), dtype=np.uint8)
        for m in masks:
            if m.dtype != np.uint8:
                m = (m > thr).astype(np.uint8)
            
            if full_mask.shape != m.shape:
                if full_mask.shape == m.T.shape:
                    m = m.T
                else:
                    raise ValueError(
                        f"[render_sam] mask shape mismatch: full_mask={full_mask.shape}, m={m.shape}"
                    )
            full_mask = np.logical_or(full_mask, m).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")

    mask_3c = np.repeat(full_mask[..., None], 3, axis=-1)

    if mode == "black":
        out = np.where(mask_3c == 1, img, 0)
    elif mode == "blur":
        blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        out = np.where(mask_3c == 1, img, blurred)
    elif mode == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray3c = np.stack([gray]*3, axis=-1)
        out = np.where(mask_3c == 1, img, gray3c)
    elif mode == "white":
        white_bg = np.ones_like(img) * 255
        out = np.where(mask_3c == 1, img, white_bg)
    elif mode == "highlight":
        blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        highlight = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        out = np.where(mask_3c == 1, highlight, blurred)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return Image.fromarray(out)


def render_depth_anything_v2_front(
        image: Image.Image,
        depth_map: np.ndarray,
        mode="gray",
        strength=1.0,
        blur_strength=25,
    ) -> Image.Image:
    """
    Render image with DepthAnything-v2 (front) depth map.
    - image: PIL Image
    - depth_map: np.ndarray (H, W)
    - mode: "gray", "black", "white", "blur", "highlight"
    - strength: blending strength for depth influence
    - blur_strength: blur kernel size (for blur/highlight)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image).astype(np.float32)
    H, W = img.shape[:2]

    # Normalize depth map to [0, 1]
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_mask = cv2.resize(depth_norm, (W, H), interpolation=cv2.INTER_CUBIC)[..., None]

    # Base gray version
    gray_target = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_target = np.stack([gray_target]*3, axis=-1).astype(np.float32)

    # Mode-dependent rendering
    if mode == "gray":
        out = img * (1 - strength * depth_mask) + gray_target * (strength * depth_mask)
    elif mode == "black":
        black_bg = np.zeros_like(img)
        out = img * (1 - depth_mask) + black_bg * depth_mask
    elif mode == "white":
        white_bg = np.ones_like(img) * 255
        out = img * (1 - depth_mask) + white_bg * depth_mask
    elif mode == "blur":
        blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        out = img * (1 - depth_mask) + blurred * depth_mask
    elif mode == "highlight":
        blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        highlight = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        out = highlight * (1 - depth_mask) + blurred * depth_mask
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def render_depth_anything_v2_back(
        image: Image.Image,
        depth_map: np.ndarray,
        strength=1.0
    ) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image).astype(np.float32)
    H, W = img.shape[:2]

    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

    depth_mask = 1.0 - depth_norm
    depth_mask = cv2.resize(depth_mask, (W, H), interpolation=cv2.INTER_CUBIC)[..., None]

    gray_target = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_target = np.stack([gray_target]*3, axis=-1).astype(np.float32)

    out = img * (1 - strength * depth_mask) + gray_target * (strength * depth_mask)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


def render_mdetr(
        image: Image.Image, 
        detections, 
        pad_ratio=0.1,
        mode="gray",
        blur_strength=25
    ) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)
    H, W = img.shape[:2]

    mask = np.zeros((H, W), dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        pad_x = int(pad_ratio * (x2 - x1))
        pad_y = int(pad_ratio * (y2 - y1))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x)
        y2 = min(H, y2 + pad_y)
        mask[y1:y2, x1:x2] = 1

    mask_3c = np.repeat(mask[..., None], 3, axis=-1)

    if mode == "black":
        out = np.where(mask_3c == 1, img, 0)
    elif mode == "blur":
        blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        out = np.where(mask_3c == 1, img, blurred)
    elif mode == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray3c = np.stack([gray]*3, axis=-1)
        out = np.where(mask_3c == 1, img, gray3c)
    elif mode == "white":
        white_bg = np.ones_like(img) * 255
        out = np.where(mask_3c == 1, img, white_bg)
    elif mode == "highlight":
        blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
        highlight = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        out = np.where(mask_3c == 1, highlight, blurred)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return Image.fromarray(out)