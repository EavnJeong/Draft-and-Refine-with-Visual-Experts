def load_grounding_dino(device="cuda"):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    model_id = "rziga/mm_grounding_dino_large_all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return {"model": model, "processor": processor}


def load_sam(device="cuda"):
    from transformers import pipeline
    mask_generator = pipeline(model="facebook/sam-vit-base", task="mask-generation")
    return {"model": mask_generator, "processor": None}


def load_depth_anything_v2(device="cuda"):
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    return {"model": model, "processor": processor}


def load_mdetr(device="cuda"):
    from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
    model_id = "SenseTime/deformable-detr"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = DeformableDetrForObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return {"model": model, "processor": processor}