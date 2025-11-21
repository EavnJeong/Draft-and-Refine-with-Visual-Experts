from .loader import load_llama3_8b, load_llama3_70b


llm_loader_map = {
    'llama-3-8b': load_llama3_8b,
    'llama-3-70b': load_llama3_70b,
}


def llm_getter(llm_name, device, cache_dir=None):
    if llm_name not in llm_loader_map:
        raise ValueError(f"LLM {llm_name} not supported.")
    print(f"🔧 Loading LLM: {llm_name}")
    model, tokenizer = llm_loader_map[llm_name](cache_dir=cache_dir, device=device)
    return model, tokenizer
    
    