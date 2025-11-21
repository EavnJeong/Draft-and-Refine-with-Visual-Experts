import torch
from typing import List, Union
from PIL import Image
import os
import re


def _llm_extract_keywords(
        q:str, 
        model, 
        tokenizer, 
        max_new_tokens:int=32) -> List[str]:
    prompt = (
        f"Extract the most important visual keywords (objects, attributes, or actions) "
        f"from the following question. Return them as a comma-separated list.\n\n"
        f"Question: {q}\n"
        f"Keywords:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    match = re.search(r"(?i)keywords:\s*(.*)", decoded)
    raw = match.group(1).strip() if match else decoded
    raw = raw.split("\n")[0]
    return [k.strip().lower() for k in raw.split(",") if k.strip()]


@torch.no_grad()
def build_query(
    model, tokenizer,
    images: List[Union[Image.Image, str]],
    questions: List[str],
    mmbench=False,
):
    results = []
    for img, q in zip(images, questions):
        save_path = str(img).replace("images", 'query_simple')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        save_path = re.sub(r'\.(jpg|png)$', '', save_path, flags=re.IGNORECASE)
        safe_q = re.sub(r'[^a-zA-Z0-9]', '_', q)
        if mmbench and len(safe_q) > 150:
            safe_q = safe_q[:60] + "_TRUNC_" + safe_q[-40:]
        save_path = f"{save_path}_{safe_q}.pb"

        if os.path.exists(save_path):
            data = torch.load(save_path)
            results.append(data)
            print(f"Loaded: {data}")
            continue

        keywords = _llm_extract_keywords(q, model, tokenizer)
        if len(keywords) == 0:
            keywords = ["object", "person", "background"]

        result = {
            "question": q,
            "query": keywords
        }
        results.append(result)
        torch.save(result, save_path)
    return results