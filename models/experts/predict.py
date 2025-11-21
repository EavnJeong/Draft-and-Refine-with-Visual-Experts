from PIL import Image
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
 

from .rendering import (
    render_grounding_dino,
    render_sam,
    render_depth_anything_v2_front,
    render_depth_anything_v2_back,
    render_mdetr,
)


@torch.no_grad()
def run_grounding_dino(
    model, 
    processor, 
    images: list[Image.Image], 
    prompts: list[str], 
    queries=None,
    device="cuda", 
    box_threshold=0.3,
    mode="gray"  # "gray", "black", "white", "highlight"
):
    def refine_detections(detections, score_threshold=0.5):
        refined_json_list, refined_text_list = [], []
        for det_list in detections:
            filtered = [d for d in det_list if d["score"] >= score_threshold]
            label_counts = Counter(d["label"] for d in filtered)
            refined_json = [{"label": k, "count": v} for k, v in label_counts.items()]
            refined_json_list.append(refined_json)
            if refined_json:
                text_desc = ", ".join([f"{v} {k}(s)" for k, v in label_counts.items()])
                context_sentence = f"The image contains {text_desc}."
            else:
                context_sentence = "No confident objects detected."
            refined_text_list.append(context_sentence)
        return refined_json_list, refined_text_list

    rendered_images, detections = [], []
    for image, prompt, query in zip(images, prompts, queries):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if len(query) == 0:
            text_input = ["detect everything"]
        else:
            text_input = query
        text_input = '. '.join(text_input)

        inputs = processor(
            images=image, 
            text=text_input, 
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)

        image_size = [(image.height, image.width)]  # (H, W)
        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            target_sizes=image_size
        )[0]

        detection = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            detection.append({
                "label": label,
                "score": round(score.item(), 3),
                "box": [round(x, 2) for x in box.tolist()]
            })

        rendered_image = render_grounding_dino(image, detection, mode=mode)
        rendered_images.append(rendered_image)
        detections.append(detection)
    return rendered_images, refine_detections(detections)


@torch.no_grad()
def run_sam(
    model, 
    processor, 
    images: list[Image.Image], 
    prompts: list[str] = None,
    queries=None,
    device="cuda",
    mode="gray"  # "gray", "black", "white", "highlight"
):
    rendered_images, detections = [], []

    for idx, image in enumerate(images):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if max(image.size) > 1024:
            scale = 1024 / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, resample=Image.BILINEAR)

        outputs = model(image)
        masks = outputs.get("masks", [])
        scores = outputs.get("scores", [])

        if masks is None or len(masks) == 0:
            rendered_images.append(image)
            detections.append([])
            continue

        mask_array = np.stack([m.astype(np.uint8) for m in masks], axis=0)

        rendered_image = render_sam(image, mask_array, mode=mode)
        rendered_images.append(rendered_image)

        detection = [
            {
                "label": "mask",
                "score": float(scores[i]) if scores is not None else None,
                "box": None
            }
            for i in range(len(masks))
        ]
        detections.append(detection)
    return rendered_images, None


@torch.no_grad()
def run_depth_anything_v2_front(
    model,
    processor,
    images: list[Image.Image],
    prompts: list[str] = None,  
    queries=None,
    device="cuda",
    strength=0.5,
    mode="gray"
):
    rendered_images, detections = [], None

    for image in images:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        outputs = model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

        depth_img = render_depth_anything_v2_front(
            image, depth_map, strength=strength, mode=mode)
        rendered_images.append(depth_img)

    return rendered_images, detections


@torch.no_grad()
def run_depth_anything_v2_back(
    model,
    processor,
    images: list[Image.Image],
    prompts: list[str] = None,  
    queries=None,
    device="cuda",
    strength=0.5,
    mode="gray"
):
    rendered_images, detections = [], None

    for image in images:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        outputs = model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

        depth_img = render_depth_anything_v2_back(image, depth_map, strength=strength)
        rendered_images.append(depth_img)

    return rendered_images, detections


@torch.no_grad()
def run_mdetr(
    model,
    processor,
    images: list[Image.Image],
    prompts: list[str],
    queries=None,
    device="cuda",
    box_threshold=0.3,
    mode="gray"  # "gray", "black", "white", "highlight"
):
    """
    Run MDETR for referring expression grounding.

    Args:
        model: MDETR (torch.nn.Module)
        processor: Hugging Face processor
        images (list): list of PIL.Image or image paths
        prompts (list[str]): referring expressions (one per image)
        device (str): "cuda" or "cpu"
        box_threshold (float): score threshold for detections

    Returns:
        rendered_images (list[PIL.Image]),
        (refined_json_list, refined_text_list)
    """

    def refine_detections(detections, score_threshold=0.5):
        refined_json_list, refined_text_list = [], []
        for det_list in detections:
            filtered = [d for d in det_list if d["score"] >= score_threshold]
            label_counts = Counter(d["label"] for d in filtered)
            refined_json = [{"label": k, "count": v} for k, v in label_counts.items()]
            refined_json_list.append(refined_json)
            if refined_json:
                text_desc = ", ".join([f"{v} {k}(s)" for k, v in label_counts.items()])
                context_sentence = f"The image contains {text_desc}."
            else:
                context_sentence = "No confident objects detected."
            refined_text_list.append(context_sentence)
        return refined_json_list, refined_text_list

    rendered_images, detections = [], []
    for image, prompt in zip(images, prompts):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(device)  # (H, W)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=box_threshold
        )[0]

        detection = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            detection.append({
                "label": str(label.item()), 
                "score": round(score.item(), 3),
                "box": [round(x, 2) for x in box.tolist()]
            })

        rendered_image = render_mdetr(image, detection, mode=mode)
        rendered_images.append(rendered_image)
        detections.append(detection)

    return rendered_images, refine_detections(detections)