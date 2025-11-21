import torch


def load_idefics(
    model_name: str = "HuggingFaceM4/idefics-9b-instruct",
    device: str = "cuda:0",
    cache_dir: str = None,
    dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    from transformers import IdeficsForVisionText2Text, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = IdeficsForVisionText2Text.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        **kwargs
    ).to(device)

    return model, processor


def load_paligemma(
    model_name: str = "google/paligemma-3b-mix-448",  
    device: str = "cuda:0",
    cache_dir: str = None,
    dtype: torch.dtype = torch.bfloat16,  
    **kwargs,
):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        cache_dir=cache_dir,
        **kwargs
    )
    return model, processor


def load_instructblip(
    model_name: str = "Salesforce/instructblip-vicuna-7b", 
    device: str = "cuda:0",
    cache_dir: str = None,
):
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    processor = InstructBlipProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        device_map=device,
    )
    return model, processor


def load_llava(
    model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    device: str = "cuda:0",
    cache_dir: str = None,
):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,              
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        device_map=device
    )
    return model, processor


def load_qwen25_vl(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda:0",
    cache_dir: str = None,
    dtype: torch.dtype = torch.bfloat16,
    **kwargs,  
):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device, 
        cache_dir=cache_dir,
        **kwargs
    )
    return model, processor


def load_cogvlm(
    model_name: str = "THUDM/cogvlm-chat-hf",
    device: str = "cuda:0",
    cache_dir: str = None,
    dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    """
    Load CogVLM (chat/VQA capable multimodal LLM).
    """
    from transformers import AutoModelForCausalLM, LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.5", cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
        **kwargs,
    ).to(device).eval()

    return model, tokenizer


def load_minigptv2(
    cfg_path: str = "models/vlm/MiniGPT/eval_configs/minigptv2_eval.yaml",
    device: str = "cuda:0",
    cache_dir: str = None,  # cache_dir is not used here but kept for consistency
):
    """
    Load MiniGPT-v2 model and vision processor.

    Args:
        cfg_path (str): path to configuration yaml file
        device (str): device to load model, e.g. 'cuda:0'

    Returns:
        model (torch.nn.Module): loaded MiniGPT-v2 model
        vis_processor: vision processor for preprocessing images
    """
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry

    # reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # config
    args = type("Args", (), {"cfg_path": cfg_path, "gpu_id": int(device.split(":")[-1]), "options": None})()
    cfg = Config(args)

    # model
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device).eval()

    # vision processor
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    return model, vis_processor