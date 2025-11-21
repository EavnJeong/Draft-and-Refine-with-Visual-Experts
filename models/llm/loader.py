from transformers import AutoTokenizer, AutoModelForCausalLM


def load_llama3_8b(device):
    model_id = "meta-llama/Meta-Llama-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto").to(device)
    return model, tokenizer


def load_llama3_70b(cache_dir, device):
    model_id = "meta-llama/Meta-Llama-3-70B"
    
    print(f"load model {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map=device
    )
    return model, tokenizer