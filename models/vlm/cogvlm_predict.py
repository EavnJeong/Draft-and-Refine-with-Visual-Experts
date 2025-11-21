import torch
import difflib
import torch.nn.functional as F
import re
from PIL import Image


@torch.no_grad()
def safe_generate_with_image(model, tokenizer, query, image, max_length=256):
    conv = model.build_conversation_input_ids(
        tokenizer,
        query=query,
        history=[],
        images=[image],
        template_version="vqa",
    )

    inputs = {
        "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
        "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
        "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
        "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
    }

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    images = inputs["images"]

    for _ in range(max_length):
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            images=images,
            use_cache=False,  
        )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        token_type_ids = torch.cat(
            [token_type_ids, torch.ones_like(next_token, device=token_type_ids.device) * 0], dim=-1
        )
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=attention_mask.device)], dim=-1
        )

    output_text = tokenizer.decode(input_ids[0][conv["input_ids"].shape[0]:], skip_special_tokens=True)
    return output_text.strip()


@torch.no_grad()
def infer_cogvlm(
        model,
        tokenizer,
        batch,
        max_length=2048,
        return_inputs=False,
        return_logits=False,
    ):
    """
    CogVLM inference (VQA / chat style, official & safe version).

    Args:
        model: CogVLM model (AutoModelForCausalLM from THUDM/cogvlm-chat-hf)
        tokenizer: LlamaTokenizer (Vicuna)
        batch: dict with keys
            - "images": list of str | PIL.Image | torch.Tensor
            - "questions": list of str
            - optional "context" (list of str)
            - optional "draft" (list of str)
        max_length: generation sequence length
        return_inputs: if True, return preprocessed inputs
        return_logits: if True, return raw logits from forward()
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

    results = []

    for i, (img, q) in enumerate(zip(batch["images"], batch["questions"])):
        proc_img = _process_image(img)

        q_text = q
        if "context" in batch and len(batch["context"]) > i:
            ctx = batch["context"][i]
            if ctx:
                q_text += f"\nExpert context: {ctx}"
        q_text += "\nAnswer concisely."

        if "draft" in batch and len(batch["draft"]) > i:
            q_text += (
                f"\nInitial answer: '{batch['draft'][i]}'. "
                f"Reconsider and correct if needed."
            )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        response = safe_generate_with_image(model, tokenizer, q_text, proc_img)

        if isinstance(response, tuple):
            response = response[0]

        response = str(response).strip()
        results.append(response)

    return results


@torch.no_grad()
def infer_cogvlm_vizwiz(
        model,
        tokenizer,
        batch,
        max_length=256,
        return_inputs=False,
        return_logits=False,
    ):
    """
    CogVLM inference for VizWiz (visually-impaired Q&A).
    - Adds explicit 'unanswerable' instruction for unclear images.
    - Encourages short, factual, single-phrase answers.
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

    results = []

    for i, (img, q) in enumerate(zip(batch["images"], batch["questions"])):
        proc_img = _process_image(img)

        q_text = (
            "You are helping a visually impaired person who cannot see the image.\n"
            f"Question: {q.strip()}\n"
            "If the question cannot be answered from the image or the image is unclear, "
            "respond with 'unanswerable'. Otherwise, answer briefly using one or a few words."
        )
        if "context" in batch and len(batch["context"]) > i:
            ctx = batch["context"][i]
            if ctx:
                q_text += f"\nAdditional visual context: {ctx.strip()}"
        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"\nPrevious answer attempt: '{draft}'. "
                    "Revise if it seems incorrect based on the image."
                )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }
        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        response = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(response, tuple):
            response = response[0]

        response = str(response).strip()
        for prefix in ["Answer:", "Response:", "Assistant:", "It shows"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        if not response or len(response) < 2:
            response = "unanswerable"

        results.append(response)

    return results


@torch.no_grad()
def infer_cogvlm_gqa(
        model,
        tokenizer,
        batch,
        max_length=256,
        return_inputs=False,
        return_logits=False,
    ):
    """
    CogVLM inference for GQA dataset.
    - Focuses on factual and reasoning-based visual questions.
    - Encourages concise (1–3 word) answers.
    - Designed for deterministic evaluation (no stylistic variation).
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

    results = []

    for i, (img, q) in enumerate(zip(batch["images"], batch["questions"])):
        proc_img = _process_image(img)

        q_text = (
            f"Question: {q.strip()}\n"
            "Look at the image carefully and answer concisely (1–3 words).\n"
            "Focus on logical reasoning and visual evidence only.\n"
            "Do not explain or justify."
        )
        if "context" in batch and len(batch["context"]) > i:
            ctx = batch["context"][i]
            if ctx:
                q_text += f"\nVisual context: {ctx.strip()}"
        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"\nInitial answer: '{draft}'. "
                    "If correct, keep it; otherwise revise concisely."
                )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }
        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        response = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(response, tuple):
            response = response[0]
        response = str(response).strip()

        for prefix in ["Answer:", "Response:", "Assistant:", "The answer is", "It is"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        if not response or len(response) < 1:
            response = "unknown"

        results.append(response)

    return results


@torch.no_grad()
def infer_cogvlm_textvqa(
        model,
        tokenizer,
        batch,
        max_length=256,
        return_inputs=False,
        return_logits=False,
    ):
    """
    Optimized CogVLM inference for TextVQA (scene-text question answering).
    Improvements:
        - Stronger text-grounding constraint ("answer ONLY using visible text")
        - Explicit numeric/symbol inclusion
        - Better OCR text/token integration (natural language form)
        - More effective draft refinement instruction
        - Less aggressive filtering
        - Extended word limit (up to 10 words)
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

    results = []

    ocr_texts = batch.get("ocr_text", None)
    ocr_tokens = batch.get("ocr_tokens", None)
    draft_answers = batch.get("draft", None)

    for i, (img, q) in enumerate(zip(batch["images"], batch["questions"])):
        proc_img = _process_image(img)

        q_text = (
            "You are a visual question answering expert specialized in reading text in images.\n"
            "Answer ONLY using text that is visibly written in the image, including numbers and symbols.\n"
            "Do not guess or infer based on context or appearance.\n"
            "If the answer cannot be read, respond exactly with 'unanswerable'.\n"
            f"Question: {q.strip()}\n"
        )

        if ocr_texts is not None and len(ocr_texts) > i and ocr_texts[i]:
            q_text += f"Visible text detected: {ocr_texts[i].strip()}\n"
        if ocr_tokens is not None and len(ocr_tokens) > i and ocr_tokens[i]:
            token_str = ", ".join(ocr_tokens[i][:30])
            q_text += f"Visible words include: {token_str}.\n"

        q_text += "Provide a concise, factual answer (1–10 words).\n"

        if draft_answers is not None and len(draft_answers) > i and draft_answers[i]:
            draft = draft_answers[i]
            if draft:
                q_text += (
                    f"Previous answer attempt: '{draft}'. "
                    "Check this carefully. If it contradicts the visible text, correct it. "
                    "Otherwise, restate the exact readable answer.\n"
                )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        response = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(response, tuple):
            response = response[0]
        response = str(response).strip()

        for prefix in ["Answer:", "Response:", "Assistant:", "It shows", "Output:"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        response_lower = response.lower()
        bad_tokens = ["", "none", "unknown", "no idea", "error"]
        if response_lower in bad_tokens or len(response) < 2:
            response = "unanswerable"

        if len(response.split()) > 10:
            response = " ".join(response.split()[:10])

        results.append(response.strip())

    return results


@torch.no_grad()
def infer_cogvlm_ocrvqa(
        model,
        tokenizer,
        batch,
        max_length=2048,
        return_inputs=False,
        return_logits=False,
    ):
    """
    Inference for CogVLM on OCR-VQA (scene-text based QA).
    Includes OCR tokens as contextual text before the question.

    Args:
        model: CogVLM model (AutoModelForCausalLM from THUDM/cogvlm-chat-hf)
        tokenizer: LlamaTokenizer (Vicuna-based tokenizer)
        batch: dict with keys
            - "images": list of str | PIL.Image | torch.Tensor
            - "questions": list of str
            - optional "ocr_token": list of list[str] or list[str]
        max_length: generation sequence length
        return_inputs: if True, return tokenized inputs (no inference)
        return_logits: if True, return raw logits
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

    def safe_generate_with_image(model, tokenizer, query, image, max_length=256):
        """Token-by-token generation without crashing."""
        conv = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=[],
            images=[image],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        images = inputs["images"]

        for _ in range(max_length):
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                images=images,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            token_type_ids = torch.cat(
                [token_type_ids, torch.zeros_like(next_token, device=token_type_ids.device)], dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=attention_mask.device)], dim=-1
            )

        output_text = tokenizer.decode(input_ids[0][conv["input_ids"].shape[0]:], skip_special_tokens=True)
        return output_text.strip()

    results = []

    for i, (img, q) in enumerate(zip(batch["images"], batch["questions"])):
        proc_img = _process_image(img)

        # --- OCR context integration ---
        ocr_tokens = batch.get("ocr_token", None)
        ocr_context = ""
        if ocr_tokens is not None and len(ocr_tokens) > i:
            tokens = ocr_tokens[i]
            if isinstance(tokens, list):
                tokens = " ".join(tokens)
            if tokens.strip():
                ocr_context = f"The image contains the following text: {tokens}\n"

        q_text = ocr_context + f"Question: {q}\nAnswer concisely based on the image and visible text."

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        response = safe_generate_with_image(model, tokenizer, q_text, proc_img)

        if isinstance(response, tuple):
            response = response[0]

        response = str(response).strip()
        results.append(response)

    return results


@torch.no_grad()
def infer_cogvlm_cococaption(
    model,
    tokenizer,
    batch,
    max_length=128,
    return_inputs=False,
    return_logits=False,
):
    """
    CogVLM inference for COCO Captioning.
    - Uses short, descriptive caption prompts.
    - Removes conversational artifacts (e.g., "Assistant:", "Sure, here is ...").
    - Prioritizes concise, factual captions.
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

    results = []

    for i, img in enumerate(batch["images"]):
        proc_img = _process_image(img)

        q_text = (
            "Describe this image in one clear, factual sentence.\n"
            "Do not start with 'This image shows' or 'A picture of'."
        )

        if "context" in batch and len(batch["context"]) > i:
            ctx = batch["context"][i]
            if ctx:
                q_text += f"\nVisual hints: {ctx.strip()}"

        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += f"\nPrevious caption: '{draft}'. Refine if inaccurate."

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        caption = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(caption, tuple):
            caption = caption[0]
        caption = str(caption).strip()

        for prefix in [
            "caption:", "description:", "this image shows", "a picture of", 
            "assistant:", "the image shows", "it shows"
        ]:
            if caption.lower().startswith(prefix):
                caption = caption[len(prefix):].strip()
        caption = caption.strip(' .')
        if not caption:
            caption = "unknown"
        results.append(caption)
    return results


@torch.no_grad()
def infer_cogvlm_nocaps(
        model,
        tokenizer,
        batch,
        max_length=128,
        return_inputs=False,
        return_logits=False,
    ):
    """
    CogVLM inference for NoCaps dataset.
    - Generates concise, factual English captions (for COCO-style metrics).
    - Removes conversational patterns ("Answer:", "This image shows...", etc.).
    - Compatible with multi-reference evaluation (CIDEr, SPICE, etc.).
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

    results = []

    for i, img in enumerate(batch["images"]):
        proc_img = _process_image(img)

        q_text = (
            "Describe the image in one short, factual English sentence.\n"
            "Avoid conversational words like 'This image shows' or 'The picture depicts'.\n"
            "Focus only on what is visible."
        )
        if "context" in batch and len(batch["context"]) > i:
            ctx = batch["context"][i]
            if ctx:
                q_text += f"\nAdditional context: {ctx.strip()}"

        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"\nPrevious caption attempt: '{draft}'. "
                    "Refine it for clarity and precision."
                )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        response = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(response, tuple):
            response = response[0]

        response = str(response).strip()

        cleanup_prefixes = [
            "Answer:", "Response:", "Caption:", "This image shows", 
            "The picture shows", "It shows", "This is", "An image of", "A photo of"
        ]
        for prefix in cleanup_prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        if not response or len(response) < 3:
            response = "A photo showing an object."

        if not response.endswith("."):
            response += "."

        results.append(response)

    return results


@torch.no_grad()
def infer_cogvlm_flickr(
    model,
    tokenizer,
    batch,
    max_length=128,
    return_inputs=False,
    return_logits=False,
):
    """
    CogVLM inference for COCO Captioning.
    - Uses short, descriptive caption prompts.
    - Removes conversational artifacts (e.g., "Assistant:", "Sure, here is ...").
    - Prioritizes concise, factual captions.
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

    results = []

    for i, img in enumerate(batch["images"]):
        proc_img = _process_image(img)

        q_text = (
            "Describe this image in one clear, factual sentence.\n"
            "Do not start with 'This image shows' or 'A picture of'."
        )
        if "context" in batch and len(batch["context"]) > i:
            ctx = batch["context"][i]
            if ctx:
                q_text += f"\nVisual hints: {ctx.strip()}"
        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += f"\nPrevious caption: '{draft}'. Refine if inaccurate."

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        caption = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(caption, tuple):
            caption = caption[0]
        caption = str(caption).strip()

        for prefix in [
            "caption:", "description:", "this image shows", "a picture of", 
            "assistant:", "the image shows", "it shows"
        ]:
            if caption.lower().startswith(prefix):
                caption = caption[len(prefix):].strip()
        caption = caption.strip(' .')
        if not caption:
            caption = "unknown"
        results.append(caption)
    return results


@torch.no_grad()
def infer_cogvlm_vcr(
    model,
    tokenizer,
    batch,
    max_length=256,
    return_inputs=False,
    return_logits=False,
):
    """
    CogVLM inference for Visual Commonsense Reasoning (VCR)
    with ViCor-style numeric output format.
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

    def _clean(t):
        if "Assistant:" in t:
            t = t.split("Assistant:")[-1]
        return t.replace("<end_of_utterance>", "").strip(" .,:;!?\"'\n\t")

    def _find_most_similar(text, choices):
        """Fallback matching by string similarity."""
        text = text.lower().strip()
        scores = [difflib.SequenceMatcher(None, text, c.lower()).ratio() for c in choices]
        return int(torch.tensor(scores).argmax().item())

    preds = []

    images = batch["images"]
    questions = batch["questions"]
    answers = batch["answer_choices"]
    rationales = batch["rationales"]
    drafts = batch.get("draft", None)

    for i, q in enumerate(questions):
        img = _process_image(images[i])

        # Build prompt with both answer/rationale sets
        ans_text = "\n".join([f"({j+1}) {a}" for j, a in enumerate(answers[i])])
        rat_text = "\n".join([f"({j+1}) {r}" for j, r in enumerate(rationales[i])])

        q_text = (
            "You are a visual reasoning assistant.\n"
            f"Question: {q.strip()}\n"
            f"Answer Choices:\n{ans_text}\n"
            f"Rationale Choices:\n{rat_text}\n"
            "Choose the best answer and rationale concisely.\n"
            "Respond in this exact format:\n"
            "The best answer is (X) because (Y).\n"
            "End your response with <end_of_utterance>.\n"
        )

        if drafts is not None and len(drafts) > i and drafts[i]:
            q_text += (
                f"The previous answer was '{drafts[i]}'. "
                "Reconsider and correct if necessary.\n"
            )
        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            preds.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            preds.append(outputs.logits)
            continue

        response = safe_generate_with_image(
            model, tokenizer, q_text, img, max_length=max_length
        )
        if isinstance(response, tuple):
            response = response[0]
        decoded = str(response).strip()
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

        ans_idx = max(0, min(len(answers[i]) - 1, ans_idx))
        rat_idx = max(0, min(len(rationales[i]) - 1, rat_idx))

        ans_text = answers[i][ans_idx] if answers[i] else "N/A"
        rat_text = rationales[i][rat_idx] if rationales[i] else "N/A"

        preds.append(f"Answer {ans_idx + 1}: {ans_text} | Rationale {rat_idx + 1}: {rat_text}")
    return preds


@torch.no_grad()
def infer_cogvlm_vsr(
    model,
    tokenizer,
    batch,
    max_length=256,
    return_inputs=False,
    return_logits=False,
):
    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        if isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(img)}")

    results = []
    # use collated keys
    for img, q in zip(batch["images"], batch["questions"]):
        proc_img = _process_image(img)
        q_text = f"Question: {q.strip()}\nAnswer only with 'Yes' or 'No'.\nDo not explain or justify."

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )
        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        resp = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(resp, tuple):
            resp = resp[0]
        resp = str(resp).strip()

        for prefix in ["Answer:", "Response:", "Assistant:", "The answer is", "It is"]:
            if resp.lower().startswith(prefix.lower()):
                resp = resp[len(prefix):].strip()

        low = resp.lower()
        if low.startswith("yes"):
            resp = "Yes"
        elif low.startswith("no"):
            resp = "No"
        elif "true" in low:
            resp = "Yes"
        elif "false" in low:
            resp = "No"
        else:
            resp = "No"

        results.append(resp)

    return results


@torch.no_grad()
def infer_cogvlm_okvqa(
        model,
        tokenizer,
        batch,
        max_length=256,
        return_inputs=False,
        return_logits=False,
    ):
    """
    CogVLM inference for OK-VQA dataset.
    - Combines visual understanding with commonsense/world knowledge.
    - Encourages short factual answers (1–5 words).
    - Supports expert context and draft refinement.
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

    results = []

    for i, (img, q) in enumerate(zip(batch["images"], batch["questions"])):
        proc_img = _process_image(img)

        q_text = (
            "You are answering a visual question that may require both image understanding "
            "and world knowledge or commonsense reasoning.\n"
            f"Question: {q.strip()}\n"
            "Use the visible evidence and what you know about the world "
            "to provide a short, factual answer (1–5 words)."
        )

        if "context" in batch and len(batch["context"]) > i:
            ctx = batch["context"][i]
            if ctx:
                q_text += f"\nHelpful background information: {ctx.strip()}"

        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"\nPrevious tentative answer: '{draft}'. "
                    "Reconsider and correct if inconsistent with the image or knowledge."
                )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        response = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(response, tuple):
            response = response[0]

        response = str(response).strip()

        for prefix in ["Answer:", "Response:", "Assistant:", "It is", "The answer is"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        low = response.lower()
        if low in ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]:
            response = "unanswerable"

        if len(response.split()) > 6:
            response = " ".join(response.split()[:6])

        if low in ["yeah", "yep", "affirmative", "correct"]:
            response = "yes"
        elif low in ["nope", "negative", "incorrect"]:
            response = "no"

        results.append(response.strip())

    return results


@torch.no_grad()
def infer_cogvlm_aokvqa(
    model,
    tokenizer,
    batch,
    mode="MC",  # "DA" for direct answer, "MC" for multiple-choice
    max_length=256,
    return_inputs=False,
    return_logits=False,
):
    """
    CogVLM inference for A-OKVQA (Augmented OK-VQA).
    - Supports both Direct-Answer (DA) and Multiple-Choice (MC) modes.
    - MC: Returns the *text* of the selected choice (not the letter).
    - DA: Returns a short factual text answer (1–5 words).
    - Compatible with DnR (context/draft fields).
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

    results = []

    for i, img in enumerate(batch["images"]):
        proc_img = _process_image(img)
        q = batch["questions"][i].strip()

        if mode == "MC":
            # --- Multiple-choice prompt ---
            choices = batch["choices"][i]
            choice_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(choices)])
            q_text = (
                "You are answering a visual question that may require both image understanding "
                "and external world knowledge.\n"
                f"Question: {q}\n"
                f"Choices:\n{choice_text}\n"
                "Look carefully at the image and respond with the correct *answer text*, "
                "not just the letter."
            )
        else:
            # --- Direct-answer prompt ---
            q_text = (
                "You are answering a visual question that may require both image understanding "
                "and commonsense/world knowledge.\n"
                f"Question: {q}\n"
                "Give a short, factual answer (1–5 words)."
            )

        if "context" in batch and len(batch["context"]) > i:
            ctx = batch["context"][i]
            if ctx:
                q_text += f"\nHelpful background info: {ctx.strip()}"
        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += f"\nInitial answer: '{draft}'. Revise if necessary."

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # --- Generate output ---
        response = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(response, tuple):
            response = response[0]
        response = str(response).strip()

        # --- Clean prefixes ---
        for prefix in ["Answer:", "Response:", "Assistant:", "The answer is", "It is"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # --- Mode-specific postprocessing ---
        if mode == "MC":
            # Try to extract choice letter (A–E)
            letter_match = re.search(r"\b([A-Ea-e])\b", response)
            if letter_match:
                letter = letter_match.group(1).upper()
                idx = ord(letter) - 65
                if 0 <= idx < len(batch["choices"][i]):
                    response = batch["choices"][i][idx]
                else:
                    response = "unknown"
            else:
                # Try to directly match choice text
                lower_choices = [c.lower() for c in batch["choices"][i]]
                resp_lower = response.lower()
                matched = None
                for c in lower_choices:
                    if c in resp_lower:
                        matched = c
                        break
                response = matched if matched else response
        else:
            # Direct answer normalization
            low = response.lower()
            if low in ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]:
                response = "unanswerable"
            if len(response.split()) > 6:
                response = " ".join(response.split()[:6])
            if low in ["yeah", "yep", "affirmative", "correct"]:
                response = "yes"
            elif low in ["nope", "negative", "incorrect"]:
                response = "no"

        results.append(response.strip())

    return results


@torch.no_grad()
def infer_cogvlm_sqa(
    model,
    tokenizer,
    batch,
    max_new_tokens=5,
    return_inputs=False,
    return_logits=False,
):
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

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    choices = batch["choices"]
    contexts = batch.get("context", None)
    lectures = batch.get("lecture", None)
    drafts = batch.get("draft", None)

    results = []

    for i, (img, q, chs) in enumerate(zip(images, questions, choices)):
        choice_str = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(chs)])

        prompt = ""
        if lectures and i < len(lectures) and lectures[i]:
            prompt += f"Lecture:\n{lectures[i].strip()}\n"
        if contexts and i < len(contexts) and contexts[i]:
            prompt += f"Context:\n{contexts[i].strip()}\n"

        prompt += (
            f"Question: {q.strip()}\n"
            f"Choices:\n{choice_str}\n"
            "Please answer with only one letter (A, B, C, D, or E).\n"
            "Answer:"
        )
        if drafts and i < len(drafts) and drafts[i]:
            prompt += f" (Initial guess: '{drafts[i]}', correct it if needed.)"

        response = safe_generate_with_image(
            model,
            tokenizer,
            query=prompt,
            image=img,
            max_length=max_new_tokens,
        )

        response = re.sub(r"[^A-Za-z]", " ", response)
        matches = re.findall(r"\b[A-Ea-e]\b", response)
        if matches:
            response = matches[0].upper()
        else:
            response = "unknown"

        results.append(response)

    return results


@torch.no_grad()
def infer_cogvlm_mme(
        model,
        tokenizer,
        batch,
        max_length=80,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for CogVLM on MME Benchmark (Yes/No tasks).
    Ensures strictly 'Yes' or 'No' answers based on visual evidence.
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

    def _postprocess_outputs(texts):
        results = []
        for t in texts:
            t = t.strip()
            for p in ["Assistant:", "Answer:", "Response:"]:
                if t.lower().startswith(p.lower()):
                    t = t[len(p):].strip()
            t = t.lower()

            # clean yes/no classification
            if "yes" in t and "no" not in t:
                results.append("Yes")
            elif "no" in t and "yes" not in t:
                results.append("No")
            else:
                results.append("")  # unclear / invalid
        return results

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    tasks = batch.get("tasks", None)

    results = []
    input_list = []

    for i, (img, q) in enumerate(zip(images, questions)):
        task_name = tasks[i] if tasks is not None and len(tasks) > i else "general"

        q_text = (
            f"Task: {task_name}\n"
            f"Question: {q.strip()}\n"
            "Answer strictly with 'Yes' or 'No' based on the image."
        )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            input_list.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        images_ = inputs["images"]

        generated = []
        for _ in range(max_length):
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                images=images_,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1).unsqueeze(-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            token_type_ids = torch.cat(
                [token_type_ids, torch.zeros_like(next_token, device=token_type_ids.device)], dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=attention_mask.device)], dim=-1
            )
            generated.append(next_token)

        if len(generated) == 0:
            decoded = ""
        else:
            decoded = tokenizer.decode(
                torch.cat(generated, dim=-1)[0],
                skip_special_tokens=True,
            )

        decoded = decoded.strip()
        results.append(decoded)

    if return_inputs:
        return input_list
    elif return_logits:
        return results
    else:
        return _postprocess_outputs(results)


@torch.no_grad()
def infer_cogvlm_mmbench(
        model,
        tokenizer,
        batch,
        max_length=80,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for CogVLM on MMBench Benchmark (Multiple-choice tasks).
    - Presents question + choices (A/B/C/D) in prompt.
    - Forces deterministic single-letter (A/B/C/D) answer output.
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

    def _postprocess_outputs(texts, choices_list):
        results = []
        for t, choices in zip(texts, choices_list):
            t = t.strip()

            # remove leading tokens
            for prefix in ["Assistant:", "Answer:", "Response:"]:
                if t.lower().startswith(prefix.lower()):
                    t = t[len(prefix):].strip()

            # Normalize spacing and case
            t_clean = t.upper().replace(".", "").strip()

            # Extract letter answer (A/B/C/D)
            found = None
            for c in ["A", "B", "C", "D"]:
                if t_clean.startswith(c) or f" {c}" in t_clean:
                    found = c
                    break

            if found is not None:
                # Optional: attach text choice for readability
                mapped = next((ch for ch in choices if ch.startswith(found)), found)
                results.append(mapped)
            else:
                results.append("")  # fallback

        return results

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    choices_list = batch["choices"]
    tasks = batch.get("tasks", None)

    results = []
    input_list = []

    for i, (img, q, choices) in enumerate(zip(images, questions, choices_list)):
        task_name = tasks[i] if tasks is not None and len(tasks) > i else "MMBench"

        choice_str = "\n".join(choices)
        q_text = (
            f"Task: {task_name}\n"
            f"Question: {q.strip()}\n"
            f"{choice_str}\n"
            "Select the single best answer by replying only with one letter (A, B, C, or D)."
        )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            input_list.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        images_ = inputs["images"]

        generated = []
        for _ in range(max_length):
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                images=images_,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1).unsqueeze(-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            token_type_ids = torch.cat(
                [token_type_ids, torch.zeros_like(next_token, device=token_type_ids.device)], dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=attention_mask.device)], dim=-1
            )
            generated.append(next_token)

        if len(generated) == 0:
            decoded = ""
        else:
            decoded = tokenizer.decode(
                torch.cat(generated, dim=-1)[0],
                skip_special_tokens=True,
            )

        decoded = decoded.strip()
        results.append(decoded)

    if return_inputs:
        return input_list
    elif return_logits:
        return results
    else:
        return _postprocess_outputs(results, choices_list)


@torch.no_grad()
def infer_cogvlm_seedbench(
        model,
        tokenizer,
        batch,
        max_length=80,
        return_logits=False,
        return_inputs=False,
    ):
    """
    CogVLM inference for SEED-Bench (multiple-choice tasks).
    Enhanced prompt to enforce structured reasoning and strict letter-only output.
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

    def _postprocess_outputs(texts):
        results = []
        for t in texts:
            t = t.strip().upper()
            for p in ["ASSISTANT:", "ANSWER:", "RESPONSE:", "CHOICE:"]:
                if t.startswith(p):
                    t = t[len(p):].strip()

            for ch in ["A", "B", "C", "D"]:
                if any(x in t for x in [f"OPTION {ch}", f"ANSWER {ch}", f" {ch} ", f"IS {ch}", f"({ch})", f"{ch}.", f"THE CORRECT ANSWER IS {ch}"]):
                    results.append(ch)
                    break
            else:
                if t in ["A", "B", "C", "D"]:
                    results.append(t)
                else:
                    results.append("")  # invalid
        return results

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    choices_list = batch["choices"]
    tasks = batch.get("question_types", None)

    results = []
    input_list = []

    for i, (img, q, choices) in enumerate(zip(images, questions, choices_list)):
        qtype = tasks[i] if tasks is not None and len(tasks) > i else "General"

        choice_text = "\n".join(choices)

        q_text = (
            f"You are a vision-language reasoning assistant.\n"
            f"Analyze the image carefully and answer the multiple-choice question below.\n"
            f"---\n"
            f"Question Type: {qtype}\n"
            f"Question: {q.strip()}\n"
            f"{choice_text}\n"
            f"---\n"
            "Think step-by-step about what you see in the image and what the question asks.\n"
            "Then, decide which option (A, B, C, or D) is most correct.\n"
            "Finally, respond in this format exactly:\n\n"
            "Answer: <LETTER>\n\n"
            "For example:\n"
            "Answer: B\n"
            "---\n"
            "Now provide your answer for the current question.\n"
            "Answer:"
        )

        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            input_list.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        images_ = inputs["images"]

        generated = []
        for _ in range(max_length):
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                images=images_,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1).unsqueeze(-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            token_type_ids = torch.cat(
                [token_type_ids, torch.zeros_like(next_token, device=token_type_ids.device)], dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=attention_mask.device)], dim=-1
            )
            generated.append(next_token)

        if len(generated) == 0:
            decoded = ""
        else:
            decoded = tokenizer.decode(
                torch.cat(generated, dim=-1)[0],
                skip_special_tokens=True,
            )

        results.append(decoded.strip())

    if return_inputs:
        return input_list
    elif return_logits:
        return results
    else:
        return _postprocess_outputs(results)


@torch.no_grad()
def infer_cogvlm_haloquest(
        model,
        tokenizer,
        batch,
        max_length=256,
        return_inputs=False,
        return_logits=False,
    ):
    """
    Inference function for CogVLM on HaloQuest dataset.

    - Uses VQA-style conversation format.
    - Returns concise factual answers.
    - Compatible with safe_generate_with_image() defined above.

    Args:
        model: CogVLM model (e.g., THUDM/cogvlm-chat-hf)
        tokenizer: corresponding tokenizer (Vicuna-based)
        batch: dict with keys:
            - "images": list of str | PIL.Image | torch.Tensor
            - "questions": list of str
        max_length: max generation length
        return_inputs: if True, return model-ready inputs
        return_logits: if True, return raw logits
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

    results = []
    images = batch["images"]
    questions = batch["questions"]

    # Optional extra info
    hallucination_types = batch.get("hallucination_type", [None] * len(images))
    image_types = batch.get("image_type", [None] * len(images))

    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        # Build prompt text
        q_text = q.strip()
        if hallucination_types[i]:
            q_text = f"[{hallucination_types[i]}] {q_text}"
        if image_types[i]:
            q_text += f"\n(Image type: {image_types[i]})"
        q_text += "\nAnswer briefly and factually."

        # Preprocess conversation
        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # Use safe sequential generation
        response = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(response, tuple):
            response = response[0]

        response = str(response).strip()
        results.append(response)

    return results


@torch.no_grad()
def infer_cogvlm_mmhalbench(
    model,
    tokenizer,
    batch,
    max_length=80,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for CogVLM on MMHal-Bench dataset.
    - Task: Visual grounding + hallucination suppression.
    - No external knowledge or imagination allowed.
    - If unclear or not visible, must respond 'uncertain' or 'not visible'.
    """

    def _process_image(img):
        """Convert image of any supported type to RGB PIL."""
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

    results = []

    for i, img in enumerate(batch["images"]):
        proc_img = _process_image(img)
        q = batch["questions"][i].strip()

        # --- Strict grounding prompt ---
        q_text = (
            "You are a visual reasoning assistant. "
            "Answer strictly based on the visible content in the image. "
            "Do NOT imagine, assume, or infer unseen details.\n\n"
            f"Question: {q}\n"
            "If the answer cannot be clearly determined from the image, respond with 'uncertain' or 'not visible'."
        )

        # --- Build CogVLM-style conversation ---
        conv = model.build_conversation_input_ids(
            tokenizer,
            query=q_text,
            history=[],
            images=[proc_img],
            template_version="vqa",
        )

        inputs = {
            "input_ids": conv["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": conv["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": conv["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[conv["images"][0].to(model.device).to(torch.bfloat16)]],
        }

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # --- Generate response ---
        response = safe_generate_with_image(model, tokenizer, q_text, proc_img, max_length=max_length)
        if isinstance(response, tuple):
            response = response[0]
        response = str(response).strip()

        # --- Clean up prefixes ---
        for prefix in ["Answer:", "Response:", "Assistant:", "The answer is", "It is"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # --- Normalize ---
        low = response.lower().strip()
        if low in ["", "unknown", "none", "error", "can't tell"]:
            response = "uncertain"
        elif len(response.split()) > 10:
            response = " ".join(response.split()[:10])  # truncate overly long outputs

        results.append(response)

    return results
