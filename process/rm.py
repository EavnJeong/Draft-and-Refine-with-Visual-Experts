import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import os
from typing import List, Union
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


@torch.no_grad()
def build_relevance_maps(
        dataset: str,
        images: List[Union[Image.Image, torch.Tensor, str]],
        questions: List[str], query_phrases: List[List[str]],
        device: str = 'cuda:0',
        num: int = 0,
        mmbench=False,
    ) -> int:

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()

    def _process_image(image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, torch.Tensor):
            if image_input.ndim == 4:
                image_input = image_input.squeeze(0)
            image = image_input.permute(1, 2, 0)
            if image.dtype.is_floating_point:
                image = (image.clamp(0, 1) * 255).byte()
            image = Image.fromarray(image.cpu().numpy()).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError("Unsupported image input type.")
        return image

    for img_path, ques, q_phrases in zip(images, questions, query_phrases):
        save_dir = os.path.dirname(img_path.replace("images", "clipseg_maps"))
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_name)[0]
        question_sanitized = re.sub(r'[^a-zA-Z0-9]', '_', ques)

        if mmbench and len(question_sanitized) > 150:  # or if mmbench == True
            question_sanitized = question_sanitized[:60] + "_TRUNC_" + question_sanitized[-40:]
        
        save_path = os.path.join(save_dir, f"{name_without_ext}_{question_sanitized}.pt")
        
        if os.path.exists(save_path):
            print(f"Exists: {save_path}, skip")
            continue

        image = _process_image(img_path)

        inputs = processor(
            text=q_phrases,
            images=[image] * len(q_phrases),
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        seg_probs = torch.sigmoid(outputs.logits) # [num_queries, H, W]

        processed = []
        for c in range(seg_probs.shape[0]):
            channel = seg_probs[c]
            # UPPER 25%
            # q75 = torch.quantile(channel, 0.75)
            # channel = torch.where(channel >= q75, channel, torch.zeros_like(channel))
            processed.append(channel)
        
        combined = torch.stack(processed, dim=0).sum(dim=0) # [H, W]
        
        combined_resized = F.interpolate(
            combined.unsqueeze(0).unsqueeze(0), 
            size=(336, 336),
            mode='bilinear',
            align_corners=False
        ).squeeze() # [336, 336]
        
        flat = combined_resized.view(-1)
        softmaxed = torch.softmax(flat, dim=0).view_as(combined_resized)

        torch.save(softmaxed.cpu(), save_path)
        print(f"Saved to {save_path}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(softmaxed.cpu().numpy(), cmap='viridis')
        plt.title("Relevance Map")
        plt.axis('off')

        plt.suptitle(f"Q: {ques}\nPhrases: {q_phrases}", fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        viz_save_dir = f'r_maps_examples/clipseg/{dataset}'
        if not os.path.exists(viz_save_dir):
            os.makedirs(viz_save_dir)
        plt.savefig(f'{viz_save_dir}/{num}.png', bbox_inches='tight')
        plt.close()
        num += 1
    return num


@torch.no_grad()
def build_relevance_maps_captioning(
    dataset: str,
    images: List[Union[Image.Image, torch.Tensor, str]],
    device: str = 'cuda:0',
    num: int = 0,
) -> int:
    """
    Build object-centric relevance maps for captioning.
    - Uses COCO object categories as textual prompts.
    - No question or query input (unlike VQA).
    - Saves both .pt map and visualization (.png).
    """

    coco_objects = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()

    def _process_image(image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, torch.Tensor):
            if image_input.ndim == 4:
                image_input = image_input.squeeze(0)
            image = image_input.permute(1, 2, 0)
            if image.dtype.is_floating_point:
                image = (image.clamp(0, 1) * 255).byte()
            image = Image.fromarray(image.cpu().numpy()).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError("Unsupported image input type.")
        return image

    for img_path in images:
        save_dir = os.path.dirname(img_path.replace("images", "clipseg_maps"))
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_name)[0]
        save_path = os.path.join(save_dir, f"{name_without_ext}_objects.pt")

        if os.path.exists(save_path):
            print(f"Exists: {save_path}, skip")
            continue

        image = _process_image(img_path)

        # Run CLIPSeg for all COCO object phrases
        inputs = processor(
            text=coco_objects,
            images=[image] * len(coco_objects),
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        seg_probs = torch.sigmoid(outputs.logits)  # [80, H, W]

        # Combine all objects (weighted sum or simple sum)
        combined = seg_probs.sum(dim=0)

        # Resize and normalize
        combined_resized = F.interpolate(
            combined.unsqueeze(0).unsqueeze(0),
            size=(336, 336),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        flat = combined_resized.view(-1)
        softmaxed = torch.softmax(flat, dim=0).view_as(combined_resized)

        torch.save(softmaxed.cpu(), save_path)
        print(f"Saved relevance map to {save_path}")

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(softmaxed.cpu().numpy(), cmap='viridis')
        plt.title("Object-Centric Relevance Map")
        plt.axis('off')

        viz_save_dir = f'r_maps_examples/clipseg/{dataset}'
        os.makedirs(viz_save_dir, exist_ok=True)
        plt.savefig(f'{viz_save_dir}/{num}.png', bbox_inches='tight')
        plt.close()
        num += 1

    return num
