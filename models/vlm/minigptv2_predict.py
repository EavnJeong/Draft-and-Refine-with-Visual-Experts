import torch
import re
import difflib
from PIL import Image
from minigpt4.conversation.conversation import Chat, Conversation, SeparatorStyle


def normalize_answer(ans: str):
    ans = ans.strip().lower()
    ans = ans.strip('"').strip("'")

    if ans.endswith("."):
        ans = ans[:-1].strip()
    if ans.startswith("yes"):
        return "yes"
    if ans.startswith("no"):
        return "no"

    digits = re.findall(r"\d+", ans)
    if digits:
        return digits[0]

    tokens = ans.split()
    if tokens:
        return tokens[0]
    return ans


@torch.no_grad()
def infer_minigptv2(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.6,
    max_new_tokens=200,
    return_inputs=False,
):
    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)

    for i, (img, q) in enumerate(zip(images, questions)):
        chat = Chat(model, vis_processor, device=device)

        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({
                "pixel_values": proc_img,   # [1, C, H, W]
                "question": q
            })
            continue

        q_lower = q.lower()
        if q_lower.startswith(("is", "are", "does", "do", "can")):
            constraint = "Answer ONLY with 'yes' or 'no'. Do not add explanation."
        elif q_lower.startswith(("how many", "number of")):
            constraint = "Answer ONLY with a single number. Do not add words."
        else:
            constraint = "Answer ONLY with a single word or short phrase."

        q_text = f"<ImageHere>\n{q}\n{constraint}"

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx is not None:
                q_text += f"\nAdditional context: {ctx}"

        if "draft" in batch and len(batch["draft"]) > i:
            q_text += (
                f"\nThe initial answer was: '{batch['draft'][i]}'. "
                f"Now, given the new image evidence, please reconsider and provide a potentially corrected answer."
            )

        conv_state = Conversation(
            system="",
            roles=(r"<s>[INST] ", r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(q_text, conv_state)
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        normalized = normalize_answer(decoded)
        results.append(normalized)

    return results


@torch.no_grad()
def infer_minigptv2_vizwiz(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1, 
    max_new_tokens=50, 
    return_inputs=False,
):
    results = []
    
    def _process_image(img):
        if isinstance(img, str): return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4: img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point: img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image): return img.convert("RGB")
        else: raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(s: str):
        if 'unanswerable' in s.lower():
            return 'unanswerable'
        return s.strip()

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)
    
    chat = Chat(model, vis_processor, device=device)

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({
                "pixel_values": proc_img,
                "question": q
            })
            continue

        system_prompt = (
            "You are a VQA assistant. Provide a single word answer based only on the image. "
            "Do not add any explanation. If the answer is not in the image, say 'unanswerable'.\n"
            "Example 1:\nQuestion: What color is the car?\nAnswer: red\n"
            "Example 2:\nQuestion: Is there a cat in the picture?\nAnswer: yes"
        )
        
        prompt_content = f"<Img><ImageHere></Img>\nQuestion: {q}\nAnswer:"
        
        if expert_context is not None and len(expert_context) > i and expert_context[i]:
            prompt_content = f"<Img><ImageHere></Img>\nContext: {expert_context[i]}\nQuestion: {q}\nAnswer:"
        
        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
        
        chat.ask(prompt_content, conv_state)
        
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )
        
        decoded = res[0] if isinstance(res, (list, tuple)) else res
        decoded = post_process(decoded)
        results.append(decoded.strip())
        
    return results


@torch.no_grad()
def infer_minigptv2_gqa(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1,
    max_new_tokens=30,
    return_inputs=False,
):
    """
    Inference for MiniGPT-v2 on GQA dataset.
    - Outputs concise, single-word or short-phrase answers.
    - If answer cannot be inferred from the image, return 'unanswerable'.
    """
    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(ans: str):
        ans = ans.strip().lower()
        if len(ans) == 0:
            return "unanswerable"
        if ans.startswith("yes"):
            return "yes"
        if ans.startswith("no"):
            return "no"
        if "unanswerable" in ans:
            return "unanswerable"
        return ans.split()[0]

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)

    chat = Chat(model, vis_processor, device=device)

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({
                "pixel_values": proc_img,
                "question": q
            })
            continue

        system_prompt = (
            "You are a precise visual question answering assistant for GQA. "
            "Answer concisely with one or few words based solely on the image. "
            "If uncertain or not visible, answer 'unanswerable'. "
            "Do NOT explain reasoning.\n"
            "Examples:\n"
            "Q: What color is the apple? A: red\n"
            "Q: Is the person holding an umbrella? A: yes\n"
            "Q: What is on the table? A: book\n"
        )

        if expert_context is not None and len(expert_context) > i and expert_context[i]:
            prompt = f"<Img><ImageHere></Img>\nContext: {expert_context[i]}\nQuestion: {q}\nAnswer:"
        else:
            prompt = f"<Img><ImageHere></Img>\nQuestion: {q}\nAnswer:"

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt, conv_state)

        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        results.append(post_process(decoded))

    return results


@torch.no_grad()
def infer_minigptv2_textvqa(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1,
    max_new_tokens=50,
    return_inputs=False,
):
    """
    Inference for MiniGPT-v2 on TextVQA (scene-text reasoning).
    - Reads and interprets visible text (signs, labels, documents, etc.).
    - Optionally uses OCR text or OCR tokens as additional context.
    - Produces concise factual answers (1–5 words).
    - Replies 'unanswerable' if the information cannot be read or found in the image.
    """

    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(s: str):
        s = s.strip()
        if not s or len(s) < 2:
            return "unanswerable"
        if any(kw in s.lower() for kw in ["unreadable", "unknown", "none", "illegible", "can't tell"]):
            return "unanswerable"
        return s

    images = batch["images"]
    questions = batch["questions"]
    ocr_texts = batch.get("ocr_text", None)
    ocr_tokens = batch.get("ocr_tokens", None)  # NEW
    draft_answers = batch.get("draft", None)

    chat = Chat(model, vis_processor, device=device)

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img, "question": q})
            continue

        system_prompt = (
            "You are an expert at reading and understanding text in images. "
            "Use only the visible words, labels, or signs to answer the question. "
            "If the text is unreadable or the answer cannot be found, reply exactly with 'unanswerable'.\n"
            "Example 1:\nQuestion: What does the sign say?\nAnswer: STOP\n"
            "Example 2:\nQuestion: What brand is shown on the cup?\nAnswer: Starbucks\n"
            "Example 3:\nQuestion: What number is visible on the jersey?\nAnswer: 24"
        )

        prompt_content = f"<Img><ImageHere></Img>\nQuestion: {q.strip()}\nAnswer:"

        context_str = ""
        if ocr_texts is not None and len(ocr_texts) > i and ocr_texts[i]:
            context_str += f"Detected text: {ocr_texts[i].strip()}"
        if ocr_tokens is not None and len(ocr_tokens) > i and ocr_tokens[i]:
            tokens_str = " | ".join(ocr_tokens[i])
            context_str += ("\n" if context_str else "") + f"Detected tokens: {tokens_str}"

        if context_str:
            prompt_content = f"<Img><ImageHere></Img>\n{context_str}\nQuestion: {q.strip()}" #\nAnswer:

        if draft_answers is not None and len(draft_answers) > i and draft_answers[i]:
            draft = draft_answers[i]
            prompt_content += (
                f"\nPrevious guess: '{draft}'. "
                "Correct it if inconsistent with the visible text."
            )
            
        prompt_content += ("\nAnswer:")

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt_content, conv_state)

        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        decoded = post_process(decoded)
        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_minigptv2_ocrvqa(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.2,
    max_new_tokens=20,
    return_inputs=False,
):
    """
    Improved inference for MiniGPT-v2 on OCR-VQA.
    - Stronger grounding on OCR text.
    - Reduces 'Yes' bias by requiring visual/textual evidence.
    - Includes diverse few-shot examples to suppress generic priors.
    """

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(s: str):
        s = s.strip()
        s = re.sub(r"^(Answer:|A:|The answer is|Based on.*?:)\s*", "", s, flags=re.IGNORECASE)
        s = s.strip(" .,:;\"'")
        if len(s.split()) > 15:
            s = re.split(r"[.,]", s)[0]
        return s.strip()

    images = batch["images"]
    questions = batch["questions"]
    ocr_tokens = batch.get("ocr_tokens", None)
    draft_answers = batch.get("draft", None)

    chat = Chat(model, vis_processor, device=device)
    results = []

    system_prompt = (
        "You are a precise OCR-based visual question answering assistant. "
        "Use ONLY the visible text and the cover image to answer. "
        "If the answer is not clearly supported by the visible text, respond 'No'. "
        "Do not hallucinate or assume based on world knowledge. "
        "Respond in one or two words only.\n\n"
        "Examples:\n"
        "Q: Who wrote this book?\nVisible text: 'By Philip Gulley'\nA: Philip Gulley\n"
        "Q: What is the title of this book?\nVisible text: 'Just Shy of Harmony'\nA: Just Shy of Harmony\n"
        "Q: What is the genre of this book?\nVisible text: 'Christian Books & Bibles'\nA: Christian Books & Bibles\n"
        "Q: Is this a children's book?\nVisible text: 'Children's Book Series'\nA: Yes\n"
        "Q: Is this a financial book?\nVisible text: 'Personal Finance Guide'\nA: Yes\n"
    )

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img, "question": q})
            continue

        # OCR text integration
        if ocr_tokens is not None and len(ocr_tokens) > i and ocr_tokens[i]:
            joined = " ".join(ocr_tokens[i][:80]).strip()
            prompt_content = (
                f"<Img><ImageHere></Img>\n"
                f"The visible text on the book cover reads:\n{joined}\n\n"
                f"Use only this text and the image to answer accurately.\n"
                f"Question: {q.strip()}\n"
                f"Answer (one or two words):"
            )
        else:
            prompt_content = (
                f"<Img><ImageHere></Img>\n"
                f"Question: {q.strip()}\nAnswer (one or two words):"
            )

        # Draft refinement (optional)
        if draft_answers is not None and len(draft_answers) > i and draft_answers[i]:
            prompt_content += f"\nPrevious guess: '{draft_answers[i]}'. Revise if it seems wrong."

        # Conversation setup
        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt_content, conv_state)

        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=1500,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        results.append(post_process(decoded))

    return results


@torch.no_grad()
def infer_minigptv2_cococaption(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.2,
    max_new_tokens=40,
    return_inputs=False,
):
    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4: img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point: img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _clean_caption(text: str) -> str:
        # Remove generic prefaces
        text = text.strip()
        for bad in [
            "caption:", "Caption:", "description:", "Description:",
            "this image", "the image", "this photo", "a photo", "an image",
            "this picture", "a picture", "I apologize", "I'm sorry"
        ]:
            text = text.replace(bad, "")
        # Cut long sentences
        text = text.strip().split(".")[0]
        text = text.strip('" ').capitalize()
        return text if text else "unknown"

    chat = Chat(model, vis_processor, device=device)
    images = batch["images"]
    for img in images:
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img})
            continue

        system_prompt = (
            "Describe the image in one short sentence (5–12 words). "
            "Do not say 'this image shows' or similar phrases."
        )
        user_prompt = "<Img><ImageHere></Img>\nDescribe briefly:"

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(user_prompt, conv_state)
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        results.append(_clean_caption(decoded))

    return results


@torch.no_grad()
def infer_minigptv2_nocaps(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.3,
    max_new_tokens=100,
    style="brief",  # or "detailed"
    return_inputs=False,
):
    """
    Caption generation for NoCaps dataset using MiniGPT-v2.
    - Generates free-form captions instead of single-word answers.
    - Does not rely on 'question' field; uses image description prompt.
    """

    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    images = batch["images"]
    expert_context = batch.get("context", None)

    chat = Chat(model, vis_processor, device=device)

    for i, img in enumerate(images):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img})
            continue

        if style == "brief":
            system_prompt = (
                "You are an image captioning assistant. "
                "Generate a short and natural caption describing the given image. "
                "Avoid long explanations or reasoning. "
                "Focus on what is visible in the image."
            )
        else:
            system_prompt = (
                "You are an image captioning assistant. "
                "Provide a detailed and descriptive caption for the image, "
                "mentioning objects, attributes, and relationships."
            )

        prompt_content = "<Img><ImageHere></Img>\nCaption:"
        if expert_context is not None and len(expert_context) > i and expert_context[i]:
            prompt_content = (
                f"<Img><ImageHere></Img>\nContext: {expert_context[i]}\nCaption:"
            )

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt_content, conv_state)
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2048,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        decoded = decoded.strip().split("\n")[0]  # take first line if multiline
        results.append(decoded)

    return results


@torch.no_grad()
def infer_minigptv2_flickr(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    max_new_tokens=25,
    temperature=0.05,
    eos_token_id=None,
    return_inputs=False,
):
    """
    Flickr caption generation using MiniGPT-v2 in pure forward mode (non-chat).
    - Direct autoregressive decoding via LLaMA forward()
    - Safe for LoRA / PEFT wrappers (no direct embed_tokens calls)
    - Produces short, factual COCO-style captions
    """

    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _clean_caption(text: str):
        text = re.sub(r"(?i)sure[^:]*:?|here'?s[^:]*:?|caption:?|description:?", "", text)
        text = text.replace('"', "").replace("'", "").strip()
        text = text.split(".")[0].strip()
        text = re.sub(r"[,;:-]\s*$", "", text)
        text = text.capitalize()
        return text if text else "Unknown scene"

    # get tokenizer and eos
    tokenizer = model.llama_tokenizer
    eos_id = eos_token_id or tokenizer.eos_token_id

    images = batch["images"]

    for img in images:
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        # prompt tokens
        prompt = (
            "Describe this image in one short factual sentence. Output only the caption."
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # initialize hidden states
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.llama_model(
                input_ids=input_ids,
                encoder_hidden_states=img_embeds,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        generated = input_ids

        if return_inputs:
            results.append({
                "pixel_values": proc_img,
                "input_ids": input_ids
            })
            continue

        # autoregressive decoding loop
        for _ in range(max_new_tokens):
            next_token_logits = outputs.logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == eos_id:
                break

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.llama_model(
                    input_ids=next_token,
                    encoder_hidden_states=img_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        caption = _clean_caption(text)
        results.append(caption)

    return results


@torch.no_grad()
def infer_minigptv2_vcr(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1,
    max_new_tokens=80,
    return_inputs=False,
):
    """
    MiniGPT-v2 inference for Visual Commonsense Reasoning (VCR)
    with ViCor-style numeric output.
    Returns [(ans_idx+1, rat_idx+1), ...] for each sample.
    """

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _clean(text):
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1]
        return text.strip(" .,:;!?\"'\n\t")

    def _find_most_similar(text, choices):
        text = text.lower().strip()
        scores = [difflib.SequenceMatcher(None, text, c.lower()).ratio() for c in choices]
        return int(torch.tensor(scores).argmax().item())

    chat = Chat(model, vis_processor, device=device)
    preds = []

    images = batch["images"]
    questions = batch["questions"]
    answers = batch["answer_choices"]
    rationales = batch["rationales"]
    drafts = batch.get("draft", None)

    for i, q in enumerate(questions):
        raw_img = _process_image(images[i])
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            preds.append({"pixel_values": proc_img, "question": q})
            continue

        # --- Build system prompt ---
        system_prompt = (
            "You are a multimodal reasoning assistant. "
            "Given an image, a question, answer choices, and rationale choices, "
            "select the best answer and rationale concisely. "
            "Respond in this exact format:\n"
            "The best answer is (X) because (Y).\n"
            "End your response with <end_of_utterance>.\n"
        )

        # --- Format textual content ---
        ans_text = "\n".join([f"({j+1}) {a}" for j, a in enumerate(answers[i])])
        rat_text = "\n".join([f"({j+1}) {r}" for j, r in enumerate(rationales[i])])

        user_prompt = (
            f"<Img><ImageHere></Img>\n"
            f"Question: {q.strip()}\n"
            f"Answer Choices:\n{ans_text}\n"
            f"Rationale Choices:\n{rat_text}\n"
        )

        if drafts is not None and len(drafts) > i and drafts[i]:
            user_prompt += (
                f"Previous answer was '{drafts[i]}'. "
                "Re-evaluate and correct if necessary.\n"
            )

        user_prompt += "Answer:"

        # --- Build conversation ---
        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
        chat.ask(user_prompt, conv_state)

        # --- Generate output ---
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        cleaned = _clean(decoded)

        # --- Extract numeric indices ---
        ans_match = re.search(r"answer\s*(?:is|:)?\s*\(?(\d+)\)?", cleaned, re.IGNORECASE)
        rat_match = re.search(r"because\s*\(?(\d+)\)?", cleaned, re.IGNORECASE)

        if ans_match:
            ans_idx = int(ans_match.group(1)) - 1
        else:
            ans_part = cleaned.split("because")[0]
            ans_idx = _find_most_similar(ans_part, answers[i])

        if rat_match:
            rat_idx = int(rat_match.group(1)) - 1
        else:
            rat_part = cleaned.split("because")[-1]
            rat_idx = _find_most_similar(rat_part, rationales[i])

        # --- Clamp to valid range ---
        ans_idx = max(0, min(len(answers[i]) - 1, ans_idx))
        rat_idx = max(0, min(len(rationales[i]) - 1, rat_idx))

        # --- Build human-readable output ---
        ans_text = answers[i][ans_idx] if answers[i] else "N/A"
        rat_text = rationales[i][rat_idx] if rationales[i] else "N/A"

        preds.append(f"Answer {ans_idx + 1}: {ans_text} | Rationale {rat_idx + 1}: {rat_text}")
    return preds


@torch.no_grad()
def infer_minigptv2_vsr(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.3,
    max_new_tokens=10,
    return_inputs=False,
):
    """
    Balanced inference for MiniGPT-v2 on VSR dataset.
    - Uses neutral tone and balanced examples to avoid Yes-bias.
    - Explicitly enforces short deterministic Yes/No output.
    """

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(ans: str):
        ans = ans.lower().strip()
        if "yes" in ans or "true" in ans:
            return "yes"
        if "no" in ans or "false" in ans:
            return "no"
        return "no"  # conservative fallback

    chat = Chat(model, vis_processor, device=device)
    images, questions = batch["images"], batch["questions"]
    results = []

    system_prompt = (
        "You are a visual reasoning expert. "
        "Look at the image and answer the spatial question accurately.\n"
        "Respond ONLY with 'Yes' or 'No' — no explanations.\n"
        "Examples:\n"
        "Q: Is the man standing behind the car? A: No\n"
        "Q: Is the dog lying on the grass? A: Yes\n"
        "Q: Is the ball floating in the sky? A: No\n"
        "Q: Is the cat sitting on the sofa? A: Yes\n"
    )

    for img, q in zip(images, questions):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img, "question": q})
            continue

        prompt = f"<Img><ImageHere></Img>\nQuestion: {q.strip()}\nAnswer (Yes or No):"

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt, conv_state)

        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        results.append(post_process(decoded))

    return results


@torch.no_grad()
def infer_minigptv2_okvqa(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.3,
    max_new_tokens=20,
    return_inputs=False,
):
    """
    Inference for MiniGPT-v2 on OK-VQA dataset.
    - Combines visual evidence with commonsense/world knowledge reasoning.
    - Produces short, factual answers (1–5 words).
    - Supports optional expert context and draft refinement.
    """

    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(ans: str):
        ans = ans.strip().lower()
        if len(ans) == 0 or ans in ["unknown", "none", "no idea", "error"]:
            return "unanswerable"
        if ans.startswith("yes"):
            return "yes"
        if ans.startswith("no"):
            return "no"
        if "unanswerable" in ans:
            return "unanswerable"
        # shorten overly long answers
        words = ans.split()
        if len(words) > 6:
            ans = " ".join(words[:6])
        return ans.strip()

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)
    drafts = batch.get("draft", None)

    chat = Chat(model, vis_processor, device=device)

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({
                "pixel_values": proc_img,
                "question": q
            })
            continue

        system_prompt = (
            "You are a visual question answering assistant for OK-VQA tasks. "
            "Use both visual evidence and general world knowledge to answer concisely. "
            "Respond with a factual phrase (1–5 words). "
            "Do not explain your reasoning.\n"
            "If the image and question together are unclear, reply 'unanswerable'.\n"
            "Examples:\n"
            "Q: What tool is used to eat soup? A: spoon\n"
            "Q: What is the capital city of France? A: Paris\n"
            "Q: What color is the banana? A: yellow\n"
        )

        prompt = f"<Img><ImageHere></Img>\nQuestion: {q.strip()}\nAnswer:"
        if expert_context is not None and len(expert_context) > i and expert_context[i]:
            prompt = (
                f"<Img><ImageHere></Img>\n"
                f"Helpful knowledge: {expert_context[i]}\nQuestion: {q.strip()}\nAnswer:"
            )
        if drafts is not None and len(drafts) > i and drafts[i]:
            prompt += (
                f"\nPrevious tentative answer: '{drafts[i]}'. "
                "Revise it if inconsistent with the image or world knowledge."
            )

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt, conv_state)

        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        results.append(post_process(decoded))

    return results


@torch.no_grad()
def infer_minigptv2_aokvqa(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.3,
    max_new_tokens=40,
    mode="MC",  # "MC" (Multiple-Choice) or "DA" (Direct-Answer)
    return_inputs=False,
):
    """
    Inference for MiniGPT-v2 on A-OKVQA dataset.
    - Supports both Multiple-Choice (MC) and Direct-Answer (DA) questions.
    - If output is like 'A: bike', returns 'bike' (not 'A').
    - Uses both visual evidence and world knowledge.
    """

    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(ans: str, mode: str, choices=None):
        ans = ans.strip().lower()

        if re.match(r"^[a-e][\)\:\.\-]\s*[a-z]", ans, re.I):
            # Split on ':', ')', or '-'
            split = re.split(r"[\:\)\-]", ans, 1)
            if len(split) > 1:
                return split[1].strip()

        if mode == "MC" and choices:
            upper_ans = ans.upper()

            if upper_ans in ["A", "B", "C", "D", "E"]:
                idx = ord(upper_ans) - 65
                if 0 <= idx < len(choices):
                    return choices[idx].strip()

            for opt in choices:
                if opt.lower() in ans:
                    return opt.strip()

        if len(ans) == 0 or ans in ["unknown", "none", "no idea", "error"]:
            return "unanswerable"
        if ans.startswith("yes"):
            return "yes"
        if ans.startswith("no"):
            return "no"
        if "unanswerable" in ans:
            return "unanswerable"

        words = ans.split()
        if len(words) > 6:
            ans = " ".join(words[:6])
        return ans.strip()

    # --- Batch fields ---
    images = batch["images"]
    questions = batch["questions"]
    choices = batch.get("choices", None)
    expert_context = batch.get("context", None)
    drafts = batch.get("draft", None)

    chat = Chat(model, vis_processor, device=device)

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img, "question": q})
            continue

        if mode == "MC":
            system_prompt = (
                "You are a knowledgeable visual question answering assistant for A-OKVQA. "
                "Use both the image and commonsense knowledge to select the best answer choice. "
                "You may respond either with the letter (A–E) or the actual choice text. "
                "Do NOT explain your reasoning. "
                "If unanswerable, reply 'unanswerable'.\n"
                "Example:\n"
                "Q: What do people wear to keep their feet warm in winter?\n"
                "Choices: (A) gloves (B) socks (C) hat (D) scarf\n"
                "Answer: socks"
            )
        else:
            system_prompt = (
                "You are a knowledgeable visual question answering assistant for A-OKVQA. "
                "Use both the image and world knowledge to answer concisely. "
                "Respond with a short factual phrase (1–5 words). "
                "If uncertain, reply 'unanswerable'."
            )

        text_parts = [f"<Img><ImageHere></Img>", f"Question: {q.strip()}"]
        if expert_context and len(expert_context) > i and expert_context[i]:
            text_parts.insert(1, f"Helpful knowledge: {expert_context[i]}")
        if drafts and len(drafts) > i and drafts[i]:
            text_parts.append(
                f"Previous tentative answer: '{drafts[i]}'. "
                "Revise if inconsistent with the image or world knowledge."
            )
        if mode == "MC" and choices and len(choices) > i and isinstance(choices[i], (list, tuple)):
            opts = [f"({chr(65+j)}) {c}" for j, c in enumerate(choices[i])]
            text_parts.append("Choices: " + " ".join(opts))
            text_parts.append("Answer:")
        else:
            text_parts.append("Answer:")

        prompt = "\n".join(text_parts)

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt, conv_state)

        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        choice_set = choices[i] if (choices and len(choices) > i) else None
        results.append(post_process(decoded, mode, choice_set))

    return results


@torch.no_grad()
def infer_minigptv2_sqa(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1,
    max_new_tokens=20,
    return_inputs=False,
):
    """
    ScienceQA inference for MiniGPT-v2 (prompt-optimized version).
    - Uses few-shot examples and stricter output formatting
    - Enforces single-letter answers (A–E) or 'UNANSWERABLE'
    """

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(ans: str):
        ans = ans.strip().upper()
        if "UNANSWERABLE" in ans:
            return "UNANSWERABLE"
        for ch in ans:
            if ch in ["A", "B", "C", "D", "E"]:
                return ch
        return "UNANSWERABLE"

    # Improved system prompt
    system_prompt = (
        "You are a science question answering assistant. "
        "Read the lecture, context, and image carefully, then answer the science question. "
        "You must choose exactly one option from (A), (B), (C), (D), or (E). "
        "Do NOT explain or justify your choice. "
        "If the question cannot be answered from the given information, output 'UNANSWERABLE'.\n\n"
        "Example 1:\n"
        "Question: Which planet is known as the Red Planet?\n"
        "Choices: (A) Earth (B) Mars (C) Venus (D) Jupiter\n"
        "Answer: B\n\n"
        "Example 2:\n"
        "Question: What is the main gas in Earth's atmosphere?\n"
        "Choices: (A) Oxygen (B) Nitrogen (C) Carbon dioxide (D) Helium\n"
        "Answer: B\n\n"
        "Now answer the next question in the same format.\n"
    )

    images = batch["images"]
    questions = batch["questions"]
    choices = batch.get("choices", None)
    lectures = batch.get("lectures", None)
    contexts = batch.get("contexts", None)
    expert_context = batch.get("context", None)

    chat = Chat(model, vis_processor, device=device)
    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img, "question": q})
            continue

        # Build improved prompt
        text_parts = [f"<Img><ImageHere></Img>"]
        if lectures and len(lectures) > i and lectures[i]:
            text_parts.append(f"Lecture: {lectures[i]}")
        if contexts and len(contexts) > i and contexts[i]:
            text_parts.append(f"Context: {contexts[i]}")
        if expert_context and len(expert_context) > i and expert_context[i]:
            text_parts.append(f"Expert Context: {expert_context[i]}")
        text_parts.append(f"Question: {q}")

        if choices and len(choices) > i and isinstance(choices[i], (list, tuple)):
            opts = [f"({chr(65+j)}) {c}" for j, c in enumerate(choices[i])]
            text_parts.append("Choices: " + " ".join(opts))

        text_parts.append("Answer (choose only one letter A–E or 'UNANSWERABLE'):")
        prompt = "\n".join(text_parts)

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt, conv_state)
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=1000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        results.append(post_process(decoded))

    return results


@torch.no_grad()
def infer_minigptv2_mme(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1,
    max_new_tokens=40,
    return_inputs=False,
):
    """
    Inference for MiniGPT-v2 on MME Benchmark (Yes/No tasks).
    Produces strictly 'Yes' or 'No' answers based on visual evidence.
    """

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(ans: str):
        ans = ans.strip().lower()
        if "yes" in ans and "no" not in ans:
            return "yes"
        if "no" in ans and "yes" not in ans:
            return "no"
        return "unanswerable"

    images = batch["images"]
    questions = batch["questions"]
    tasks = batch.get("tasks", None)
    chat = Chat(model, vis_processor, device=device)
    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img, "question": q})
            continue

        task_name = tasks[i] if tasks is not None and len(tasks) > i else "General"

        system_prompt = (
            "You are a precise visual question answering assistant for the MME Benchmark. "
            "Answer strictly with either 'Yes' or 'No' based only on the image content. "
            "Do not explain your reasoning or include extra words.\n"
            "Example 1:\nQuestion: Is there a dog in the image?\nAnswer: Yes\n"
            "Example 2:\nQuestion: Is the sign written in Chinese?\nAnswer: No\n"
        )

        user_prompt = (
            f"<Img><ImageHere></Img>\n"
            f"Task: {task_name}\n"
            f"Question: {q.strip()}\n"
            f"Answer (only 'Yes' or 'No'):"
        )

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(user_prompt, conv_state)
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=512,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        results.append(post_process(decoded))

    return results


@torch.no_grad()
def infer_minigptv2_mmbench(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1,
    max_new_tokens=50,
    return_inputs=False,
):
    """
    Inference for MiniGPT-v2 on MMBench (multiple-choice tasks).
    Returns the final choice label (A/B/C/D) by matching model output with provided options.
    """

    def _process_image(img):
        """Convert any supported input type to RGB PIL image."""
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process_choice(ans: str, choices: list):
        """Map model output to one of the valid options."""
        ans_low = ans.strip().lower()
        label = "unanswerable"

        for opt in choices:
            opt_label, opt_text = opt.split(":", 1)
            if opt_label.lower() in ans_low:
                return opt_label.strip().upper()
            if opt_text.strip().lower() in ans_low:
                return opt_label.strip().upper()

        if any(x in ans_low for x in ["a", "option a"]):
            label = "A"
        elif any(x in ans_low for x in ["b", "option b"]):
            label = "B"
        elif any(x in ans_low for x in ["c", "option c"]):
            label = "C"
        elif any(x in ans_low for x in ["d", "option d"]):
            label = "D"
        return label

    images = batch["images"]
    questions = batch["questions"]
    choices_list = batch.get("choices", None)
    tasks = batch.get("tasks", None)

    chat = Chat(model, vis_processor, device=device)
    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img, "question": q})
            continue

        task_name = tasks[i] if tasks is not None and len(tasks) > i else "General"
        choices = choices_list[i] if choices_list is not None else []

        system_prompt = (
            "You are a visual QA assistant for MMBench.\n"
            "Select the best answer (A/B/C/D) using only the image and question.\n"
            "Reply with one letter only. Example: Q: What color is the cat? A: C"
        )
        # --- User prompt ---
        choice_text = "\n".join(choices) if choices else "No choices provided."
        user_prompt = (
            f"<Img><ImageHere></Img>\n"
            f"Task: {task_name}\n"
            f"Question: {q.strip()}\n"
            f"Choices:\n{choice_text}\n"
            "Answer (only one letter A/B/C/D):"
        )

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(user_prompt, conv_state)
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=512,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        final_choice = post_process_choice(decoded, choices)
        results.append(final_choice)

    return results

    
@torch.no_grad()
def infer_minigptv2_seedbench(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1,
    max_new_tokens=50,
    return_inputs=False,
):
    """
    Inference for MiniGPT-v2 on SEED-Bench (multiple-choice).
    - Handles image-based multiple-choice tasks (A–D).
    - Incorporates question type context for reasoning.
    - Returns concise answer letters ("A", "B", "C", "D", or "unanswerable").
    """

    def _process_image(img):
        """Convert any supported input type to RGB PIL image."""
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process_choice(ans: str, choices: list):
        """Map model output to one of the valid multiple-choice letters."""
        ans_low = ans.strip().lower()
        label = "unanswerable"

        # Direct match to label or text
        for opt in choices:
            if ":" not in opt:
                continue
            opt_label, opt_text = opt.split(":", 1)
            if opt_label.lower() in ans_low:
                return opt_label.strip().upper()
            if opt_text.strip().lower() in ans_low:
                return opt_label.strip().upper()

        # Fallback heuristics
        if any(x in ans_low for x in ["a", "option a"]):
            label = "A"
        elif any(x in ans_low for x in ["b", "option b"]):
            label = "B"
        elif any(x in ans_low for x in ["c", "option c"]):
            label = "C"
        elif any(x in ans_low for x in ["d", "option d"]):
            label = "D"
        return label

    # --- Unpack batch ---
    images = batch["images"]
    questions = batch["questions"]
    choices_list = batch["choices"]
    qtypes = batch.get("question_types", ["General"] * len(images))

    chat = Chat(model, vis_processor, device=device)
    results = []

    # --- Iterate through each sample ---
    for img, q, choices, qtype in zip(images, questions, choices_list, qtypes):
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({"pixel_values": proc_img, "question": q})
            continue

        # --- Build conversation ---
        system_prompt = (
            "You are a visual reasoning assistant for SEED-Bench.\n"
            "Each question provides multiple choices (A–D).\n"
            "Your task is to analyze the image and select the correct letter (A, B, C, or D).\n"
            "Respond with only the single letter. No explanations.\n"
            "Example:\nQuestion: What color is the car?\nChoices: A: Red, B: Blue, C: Green, D: Black\nAnswer: A\n"
        )

        # --- Build user prompt with question type ---
        choice_text = "\n".join(choices)
        user_prompt = (
            f"<Img><ImageHere></Img>\n"
            f"Task: {qtype}\n"
            f"Question: {q.strip()}\n"
            f"Choices:\n{choice_text}\n"
            "Answer (A/B/C/D):"
        )

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(user_prompt, conv_state)
        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=512,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        final_choice = post_process_choice(decoded, choices)
        results.append(final_choice)

    return results


@torch.no_grad()
def infer_minigptv2_haloquest(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.1,
    max_new_tokens=50,
    return_inputs=False,
):
    """
    Inference function for HaloQuest dataset using MiniGPT-v2.
    Args:
        model: MiniGPT-v2 model instance.
        vis_processor: Image preprocessor (same as in training).
        batch: Dictionary containing HaloQuest samples with keys:
               ["image_path", "question", "groundtruth", "hallucination_type", "image_type"].
        device: CUDA device.
        temperature: Generation temperature.
        max_new_tokens: Max tokens to generate.
        return_inputs: If True, return inputs instead of model outputs.
    Returns:
        List of model answers (or input dicts if return_inputs=True).
    """

    results = []

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def post_process(s: str):
        if "unanswerable" in s.lower():
            return "unanswerable"
        return s.strip()

    images = batch["images"]
    questions = batch["questions"]
    halluc_types = batch.get("hallucination_type", None)
    image_types = batch.get("image_type", None)

    chat = Chat(model, vis_processor, device=device)

    for i in range(len(images)):
        img_path = images[i]
        q = questions[i]

        raw_img = _process_image(img_path)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({
                "pixel_values": proc_img,
                "question": q,
                "image_path": img_path,
            })
            continue

        system_prompt = (
            "You are a visual question answering assistant. "
            "Answer concisely and based only on the visual evidence in the image. "
            "If the question cannot be answered from the image, respond with 'unanswerable'.\n"
            "Example 1:\nQuestion: What color is the shirt?\nAnswer: blue\n"
            "Example 2:\nQuestion: Is there a clock visible?\nAnswer: yes\n"
            "Example 3:\nQuestion: What is written on the board?\nAnswer: unanswerable"
        )

        # Add hallucination / image type context (optional, useful for analysis)
        prefix_context = ""
        if halluc_types is not None and image_types is not None:
            prefix_context = f"(Image type: {image_types[i]}, Hallucination type: {halluc_types[i]})\n"

        prompt_content = (
            f"<Img><ImageHere></Img>\n"
            f"{prefix_context}Question: {q}\nAnswer:"
        )

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(prompt_content, conv_state)

        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        decoded = post_process(decoded)
        results.append(decoded)

    return results

    
@torch.no_grad()
def infer_minigptv2_mmhalbench(
    model,
    vis_processor,
    batch,
    device="cuda:0",
    temperature=0.6,
    max_new_tokens=150,
    return_inputs=False,
):
    """
    Simplified MMHal-Bench inference for MiniGPT-v2.
    - Identical logic to original, but without normalization.
    - Prompt encourages single short factual answers.
    """
    results = []

    def _process_image(img):
        """Convert str/tensor/Image to standardized RGB PIL Image."""
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    images = batch["images"]
    questions = batch["questions"]
    qtypes = batch.get("question_types", None)
    expert_context = batch.get("context", None)

    for i, (img, q) in enumerate(zip(images, questions)):
        chat = Chat(model, vis_processor, device=device)
        raw_img = _process_image(img)
        proc_img = vis_processor(raw_img).unsqueeze(0).to(device)
        img_embeds, _ = model.encode_img(proc_img)

        if return_inputs:
            results.append({
                "pixel_values": proc_img,
                "question": q,
                "question_type": qtypes[i] if qtypes is not None else None
            })
            continue

        q_lower = q.lower()
        if q_lower.startswith(("is", "are", "does", "do", "can", "has", "have")):
            constraint = "Answer with only 'yes' or 'no'."
        elif q_lower.startswith(("how many", "number of", "count")):
            constraint = "Answer with only a number (e.g., '2')."
        else:
            constraint = "Answer with a single short factual word or phrase. No sentences."

        qtype_str = qtypes[i] if qtypes is not None and len(qtypes) > i else None
        if qtype_str:
            type_prompt = f"This is a {qtype_str}-type question. "
        else:
            type_prompt = ""

        # --- concise system prompt for single-word factual answer ---
        system_prompt = (
            "You are a precise vision-language evaluator for MMHal-Bench.\n"
            "Provide only a single short, factual answer grounded in the image.\n"
            "Do not explain or add extra words. If the answer is unclear, say 'uncertain'."
        )

        q_text = (
            f"{type_prompt}<ImageHere>\n"
            f"Question: {q.strip()}\n"
            f"{constraint}"
        )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nContext from experts: {ctx.strip()}"

        if "draft" in batch and len(batch["draft"]) > i:
            draft_ans = batch["draft"][i]
            if draft_ans:
                q_text += (
                    f"\nInitial draft answer: '{draft_ans.strip()}'. "
                    f"Respond with the corrected or confirmed final answer only."
                )

        conv_state = Conversation(
            system=system_prompt,
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat.ask(q_text, conv_state)

        res = chat.answer(
            conv=conv_state,
            img_list=[img_embeds],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=2000,
        )

        decoded = res[0] if isinstance(res, (list, tuple)) else res
        results.append(decoded.strip())

    return results
 
