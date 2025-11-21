import torch
import difflib
import re
from PIL import Image


@torch.no_grad()
def infer_idefics(
    model,
    processor,
    batch,
    max_new_tokens=100,
    return_logits=False,
    return_inputs=False,
):
    """
    batch: dict with keys
        - "images": list of str | PIL.Image | torch.Tensor
        - "questions": list of str
        - (optional) "context": list of str
        - (optional) "draft": list of str
    """

    def _process_image(img):
        if isinstance(img, str):
            return img
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

    def postprocess_idefics_outputs(texts):
        number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'")

            if t.lower().startswith("yes"):
                clean.append("Yes")
                continue
            if t.lower().startswith("no"):
                clean.append("No")
                continue

            words = t.split()[:3]
            filtered = [w for w in words if w.lower() not in ["the", "a", "an", "is"]]
            converted = [number_map.get(w.lower(), w) for w in filtered]
            t = " ".join(converted)

            clean.append(t)
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    contexts = batch.get("context", None)
    drafts = batch.get("draft", None)

    full_prompts = []
    for i, q in enumerate(questions):
        prompt = f"User: {q}"

        if contexts is not None and len(contexts) > i and contexts[i]:
            prompt += f"\nContext: {contexts[i]}"

        if drafts is not None and len(drafts) > i and drafts[i]:
            prompt += (
                f"\nThe initial answer was: '{drafts[i]}'. "
                "Now reconsider and provide the corrected answer if needed."
            )

        prompt += " Answer with ONLY one word.<end_of_utterance>\nAssistant:"
        full_prompts.append(prompt)

    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    if return_inputs:
        inputs_list = []
        for img, q in zip(images, full_prompts):
            inp = processor(
                text=q,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            inputs_list.append(inp)
        return inputs_list

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)

    results = postprocess_idefics_outputs(decoded)
    return results


@torch.no_grad()
def infer_idefics_vizwiz(
    model,
    processor,
    batch,
    max_new_tokens=100,
    return_logits=False,
    return_inputs=False,
):
    def _process_image(img):
        if isinstance(img, str):
            return img
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

    def postprocess_vizwiz_outputs(texts):
        number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'")

            if len(t) == 0:
                clean.append("unanswerable")
                continue

            low = t.lower()
            if any(kw in low for kw in ["cannot", "unable", "unanswerable", "not visible", "can't see", "too dark"]):
                clean.append("unanswerable")
                continue
            if low.startswith("yes"):
                clean.append("yes")
                continue
            if low.startswith("no"):
                clean.append("no")
                continue

            words = t.split()[:3]
            filtered = [w for w in words if w.lower() not in ["the", "a", "an", "is"]]
            converted = [number_map.get(w.lower(), w) for w in filtered]
            t = " ".join(converted)
            clean.append(t if len(t) > 0 else "unanswerable")

        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    contexts = batch.get("context", None)
    drafts = batch.get("draft", None)

    full_prompts = []
    for i, q in enumerate(questions):
        prompt = (
            f"User: {q.strip()}\n"
            "Carefully observe the image and read any visible text.\n"
            "If the question cannot be answered from the image, reply exactly with 'unanswerable'.\n"
            "Provide a short, factual answer (1–3 words).<end_of_utterance>\nAssistant:"
        )

        if contexts is not None and len(contexts) > i and contexts[i]:
            prompt += f"\nAdditional context: {contexts[i]}"

        if drafts is not None and len(drafts) > i and drafts[i]:
            prompt += (
                f"\nThe initial guess was '{drafts[i]}'. "
                "Re-evaluate and correct it if wrong."
            )

        full_prompts.append(prompt)

    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # --- Modified return_inputs block (same as infer_idefics) ---
    if return_inputs:
        inputs_list = []
        for img, q in zip(images, full_prompts):
            inp = processor(
                text=q,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            inputs_list.append(inp)
        return inputs_list
    # ------------------------------------------------------------

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)

    results = postprocess_vizwiz_outputs(decoded)
    return results


@torch.no_grad()
def infer_idefics_gqa(
    model,
    processor,
    batch,
    max_new_tokens=20,
    return_inputs=False,
):
    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)  # C, H, W -> H, W, C
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()  # [0,1] float -> [0,255] byte
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def postprocess_vqa_output(texts):
        number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        clean_texts = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]

            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'")

            if t.lower().startswith("yes"):
                clean_texts.append("Yes")
                continue
            if t.lower().startswith("no"):
                clean_texts.append("No")
                continue

            words = t.split()[:3]
            filtered_words = [w for w in words if w.lower() not in ["the", "a", "an", "is"]]
            converted_words = [number_map.get(w.lower(), w) for w in filtered_words]

            processed_t = " ".join(converted_words)
            clean_texts.append(processed_t)

        return clean_texts

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    prompts = [f"User:{q}<end_of_utterance>\nAssistant:" for q in questions]

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        input_list = []
        for img, q in zip(images, questions):
            inp = processor(
                text=f"User:{q}<end_of_utterance>\nAssistant:",
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    generated_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    results = postprocess_vqa_output(decoded_texts)

    return results


@torch.no_grad()
def infer_idefics_textvqa(
    model,
    processor,
    batch,
    max_new_tokens=80,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for Idefics-9B-Instruct on TextVQA.
    - Reads and interprets visible text within the image.
    - Encourages short, factual answers (1–3 words).
    - Returns 'unanswerable' if the text is not readable or unclear.
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

    def postprocess_textvqa_outputs(texts):
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'").lower()

            # heuristic cleaning
            if not t or len(t) < 1:
                clean.append("unanswerable")
                continue
            if any(w in t for w in ["i don't know", "not sure", "unclear", "cannot read"]):
                clean.append("unanswerable")
                continue

            # truncate long sentences → keep concise answer
            words = t.split()
            if len(words) > 3:
                t = " ".join(words[:3])
            clean.append(t)
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    ocr_tokens = batch.get("ocr_tokens", None)

    # --- Build prompts ---
    full_prompts = []
    for i, q in enumerate(questions):
        ocr_hint = ""
        if ocr_tokens is not None and len(ocr_tokens) > i:
            ocr_list = ocr_tokens[i]
            if isinstance(ocr_list, list) and len(ocr_list) > 0:
                top_ocr = ", ".join(ocr_list[:10])  # up to 10 OCR tokens
                ocr_hint = f"Visible text: {top_ocr}\n"

        prompt = (
            f"User: You are a TextVQA assistant.\n"
            f"Read the text in the image and answer the question accurately.\n"
            f"{ocr_hint}"
            f"Question: {q.strip()}\n"
            "If the answer cannot be determined from the visible text, respond exactly with 'unanswerable'.\n"
            "Answer briefly (1–3 words).<end_of_utterance>\nAssistant:"
        )
        full_prompts.append(prompt)

    # --- Preprocess for model ---
    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Optionally return raw inputs
    if return_inputs:
        input_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    # --- Generate answers ---
    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = postprocess_textvqa_outputs(decoded)
    return results


@torch.no_grad()
def infer_idefics_ocrvqa(
    model,
    processor,
    batch,
    max_new_tokens=50,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for Idefics-9B-Instruct on OCR-VQA dataset.
    - Encourages the model to read and interpret visible text in the image.
    - Answers should be short and factual (1–3 words).
    - If the answer cannot be derived from the image text, respond 'unanswerable'.
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

    def postprocess_ocrvqa_outputs(texts):
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'").lower()

            if "yes" in t and "no" not in t:
                clean.append("Yes")
                continue
            if "no" in t and "yes" not in t:
                clean.append("No")
                continue

            if len(t.split()) > 5:
                t = " ".join(t.split()[:5])
            clean.append(t.capitalize())
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]

    # Build OCR-specific prompts
    full_prompts = []
    for q in questions:
        prompt = (
            "User: You are a visual reading assistant.\n"
            "Read any visible text in the image and answer concisely.\n"
            "If the answer cannot be derived from the text, say 'unanswerable'.\n"
            f"Question: {q.strip()}\n"
            "<end_of_utterance>\nAssistant:"
        )
        full_prompts.append(prompt)

    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        input_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    # End and forbidden tokens
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = postprocess_ocrvqa_outputs(decoded)
    return results


@torch.no_grad()
def infer_idefics_cococaption(
    model,
    processor,
    batch,
    max_new_tokens=40,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for IDEFICS on COCO Captioning task.
    batch: dict with keys
        - "images": list of str | PIL.Image | torch.Tensor
    """

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def postprocess_caption_output(texts):
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = (
                t.replace("<end_of_utterance>", "")
                .replace("Caption:", "")
                .replace("Description:", "")
                .strip()
            )
            # remove quotes, punctuation
            t = t.strip(" .,:;!?\"'").capitalize()
            clean.append(t)
        return clean

    images = [_process_image(img) for img in batch["images"]]

    # --- Caption-style prompt ---
    full_prompts = [
        "User: Describe this image in one concise sentence.<end_of_utterance>\nAssistant:"
        for _ in images
    ]

    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        inputs_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            inputs_list.append(inp)
        return inputs_list

    # --- Generate captions ---
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    captions = postprocess_caption_output(decoded)
    return captions


@torch.no_grad()
def infer_idefics_nocaps(
    model,
    processor,
    batch,
    max_new_tokens=40,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for IDEFICS on NoCaps Captioning task.
    batch: dict with keys
        - "images": list of str | PIL.Image | torch.Tensor
        - "domains": list of str (optional: in-domain / near-domain / out-domain)
        - "image_ids": list of str
    """

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def postprocess_caption_output(texts):
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = (
                t.replace("<end_of_utterance>", "")
                .replace("Caption:", "")
                .replace("Description:", "")
                .strip()
            )
            t = t.strip(" .,:;!?\"'").capitalize()
            clean.append(t)
        return clean

    images = [_process_image(img) for img in batch["images"]]

    # --- Caption-style prompt ---
    full_prompts = [
        "User: Describe this image in one concise sentence.<end_of_utterance>\nAssistant:"
        for _ in images
    ]

    # --- Tokenization ---
    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # --- Return prepared inputs for debugging ---
    if return_inputs:
        inputs_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            inputs_list.append(inp)
        return inputs_list

    # --- Exit conditions ---
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    # --- Return logits (optional) ---
    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    # --- Generate captions ---
    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    captions = postprocess_caption_output(decoded)
    return captions


@torch.no_grad()
def infer_idefics_flickr(
    model,
    processor,
    batch,
    max_new_tokens=40,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for IDEFICS on Flickr Captioning task.
    - Mirrors COCO caption inference structure.
    - Uses concise English caption prompt.
    - Cleans unwanted prefixes and artifacts.
    """

    def _process_image(img):
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def postprocess_caption_output(texts):
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = (
                t.replace("<end_of_utterance>", "")
                .replace("Caption:", "")
                .replace("Description:", "")
                .strip()
            )
            t = t.strip(" .,:;!?\"'").capitalize()
            clean.append(t)
        return clean

    # --- Load and preprocess images ---
    images = [_process_image(img) for img in batch["images"]]

    # --- Prompt setup ---
    full_prompts = [
        "User: Describe this image briefly in one sentence.<end_of_utterance>\nAssistant:"
        for _ in images
    ]

    # --- Tokenization ---
    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # --- Return prepared inputs (debugging) ---
    if return_inputs:
        inputs_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            inputs_list.append(inp)
        return inputs_list

    # --- Exit conditions ---
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    # --- Return logits only ---
    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    # --- Generate captions ---
    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    captions = postprocess_caption_output(decoded)
    return captions


@torch.no_grad()
def infer_idefics_vcr(
    model,
    processor,
    batch,
    max_new_tokens=60,
    temperature=0.2,
    return_inputs=False,
    return_logits=False,
):
    """
    IDEFICS inference for VCR (QA→R) using ViCor-style reasoning prompt.
    - Unified generation: "Answer is A because B" format.
    - Encourages visual observation + commonsense reasoning.
    - Postprocessing maps natural text outputs → numeric indices (e.g., (3, 2)).
    """

    def _find_most_similar(text, choices):
        """Return most similar index based on string similarity."""
        text = text.lower().strip()
        scores = [difflib.SequenceMatcher(None, text, c.lower()).ratio() for c in choices]
        return int(torch.tensor(scores).argmax().item())

    def _process_image(img):
        """Ensure image is RGB PIL."""
        if isinstance(img, str):
            return img
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            return Image.fromarray(img.cpu().numpy()).convert("RGB")
        raise ValueError(f"Unsupported image type: {type(img)}")

    def _clean(t):
        """Clean decoding output."""
        if "Assistant:" in t:
            t = t.split("Assistant:")[-1]
        return t.replace("<end_of_utterance>", "").strip()

    preds = []

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    answers = batch["answer_choices"]
    rationales = batch["rationales"]

    # ---------- ViCor-style unified reasoning prompt ----------
    full_prompts = []
    for i, q in enumerate(questions):
        ans_text = "\n".join([f"({j+1}) {a}" for j, a in enumerate(answers[i])])
        rat_text = "\n".join([f"({j+1}) {r}" for j, r in enumerate(rationales[i])])

        prompt = (
            "You are a multimodal reasoning assistant. "
            "Observe the image carefully and reason about what is happening.\n"
            f"Question: {q.strip()}\n"
            f"Answer Choices:\n{ans_text}\n"
            f"Rationale Choices:\n{rat_text}\n"
            "First, describe briefly what you see in the image.\n"
            "Then, use that observation to decide the most plausible answer and rationale.\n"
            "Respond in this format:\n"
            "I see ... Therefore, the best answer is <chosen answer> because <chosen rationale>.\n"
            "End your response with <end_of_utterance>.\n"
            "Assistant:"
        )
        full_prompts.append(prompt)

    # ---------- Tokenize ----------
    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        inputs_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            inputs_list.append(inp)
        return inputs_list

    eos = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    # ---------- Generate ----------
    gen_ids = model.generate(
        **inputs,
        eos_token_id=eos,
        bad_words_ids=bad,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        temperature=temperature,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    cleaned = [_clean(t) for t in decoded]

    # ---------- Parse outputs + postprocess ----------
    for i, t in enumerate(cleaned):
        # Example: "I see two people arguing. Therefore, the best answer is (3) He is angry because (2) He looks upset."
        ans_match = re.search(r"answer\s*(?:is|:)?\s*\(?(\d+)\)?", t, re.IGNORECASE)
        rat_match = re.search(r"because\s*\(?(\d+)\)?", t, re.IGNORECASE)

        if ans_match:
            ans_idx = int(ans_match.group(1)) - 1
        else:
            # fallback: use textual similarity
            ans_part = t.split("because")[0]
            ans_idx = _find_most_similar(ans_part, answers[i])

        if rat_match:
            rat_idx = int(rat_match.group(1)) - 1
        else:
            rat_part = t.split("because")[-1]
            rat_idx = _find_most_similar(rat_part, rationales[i])

        ans_text = answers[i][ans_idx] if 0 <= ans_idx < len(answers[i]) else "N/A"
        rat_text = rationales[i][rat_idx] if 0 <= rat_idx < len(rationales[i]) else "N/A"

        preds.append(
            f"Answer {ans_idx + 1}: {ans_text} | Rationale {rat_idx + 1}: {rat_text}"
        )
    return preds


@torch.no_grad()
def infer_idefics_vsr(
    model, 
    processor, 
    batch, 
    max_new_tokens=20, 
    return_inputs=False
):
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
        raise ValueError(f"Unsupported image type: {type(img)}")

    def postprocess_vsr_output(texts):
        outs = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip().strip(" .,:;!?\"'")
            tl = t.lower()
            if tl.startswith("yes"): outs.append("Yes"); continue
            if tl.startswith("no"): outs.append("No"); continue
            outs.append("Yes" if "true" in tl else "No")
        return outs

    # use collated keys
    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    prompts = [
        f"User: Consider the image carefully and answer truthfully.\n"
        f"Question: Is the caption accurate?\nCaption: {q}\nAnswer with only 'Yes' or 'No'.<end_of_utterance>\nAssistant:"
        for q in questions
    ]

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        packs = []
        for img, q in zip(images, questions):
            inp = processor(
                text=f"User:{q}<end_of_utterance>\nAssistant:",
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            packs.append(inp)
        return packs

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
    )
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return postprocess_vsr_output(decoded)


@torch.no_grad()
def infer_idefics_okvqa(
    model,
    processor,
    batch,
    max_new_tokens=50,
    return_logits=False,
    return_inputs=False,
):
    """
    Optimized inference for Idefics-9B-Instruct on OK-VQA.
    Keeps original structure; only fixes image handling (<image> token issue).
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

    def postprocess_okvqa_outputs(texts):
        number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        clean = []
        for t in texts:
            t = t.split("Assistant:")[-1] if "Assistant:" in t else t
            t = t.replace("<end_of_utterance>", "").strip(" .,:;!?\"'").lower()
            t = t.replace("answer ", "")
            if not t or any(x in t for x in ["don't know", "unknown", "can't", "none"]):
                clean.append("unanswerable")
                continue
            words = t.split()[:4]
            filtered = [w for w in words if w not in ["the", "a", "an", "is", "it", "this", "that"]]
            converted = [number_map.get(w, w) for w in filtered]
            clean.append(" ".join(converted).strip() or "unanswerable")
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]

    # --- prompt stays same (no <image> token textually) ---
    prompts = []
    for q in questions:
        prompt = (
            f"User: Question: {q.strip()} "
            "Answer with only the correct word or short phrase.\nAssistant: The answer is"
        )
        prompts.append(prompt)

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        inputs_list = []
        for img, prompt in zip(images, prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            inputs_list.append(inp)
        return inputs_list

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    gen_ids = model.generate(
        **inputs,
        eos_token_id=eos_token_id,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = postprocess_okvqa_outputs(decoded)
    return results


@torch.no_grad()
def infer_idefics_aokvqa(
    model,
    processor,
    batch,
    max_new_tokens=80,
    mode="MC",  # "DA" (direct-answer) or "MC" (multiple-choice)
    return_logits=False,
    return_inputs=False,
):
    """
    Optimized inference for Idefics-9B-Instruct on A-OKVQA.
    - Handles both Direct-Answer (DA) and Multiple-Choice (MC) formats.
    - Cleans and normalizes generations for factual short answers.
    - In MC mode, returns the *text* corresponding to the chosen letter (A–E).
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

    def postprocess_aokvqa_outputs(texts):
        number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        clean = []
        for t in texts:
            t = t.split("Assistant:")[-1] if "Assistant:" in t else t
            t = t.replace("<end_of_utterance>", "").strip(" .,:;!?\"'").lower()
            t = t.replace("answer ", "")
            if not t or any(x in t for x in ["don't know", "unknown", "can't", "none"]):
                clean.append("unanswerable")
                continue
            words = t.split()[:4]
            filtered = [w for w in words if w not in ["the", "a", "an", "is", "it", "this", "that"]]
            converted = [number_map.get(w, w) for w in filtered]
            clean.append(" ".join(converted).strip() or "unanswerable")
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]

    # --- Prompt construction ---
    prompts = []
    if mode.upper() == "DA":  # Direct-answer
        for q in questions:
            prompt = (
                f"User: Question: {q.strip()} "
                "Answer with only the correct word or short phrase.\nAssistant: The answer is"
            )
            prompts.append(prompt)
    elif mode.upper() == "MC":  # Multiple-choice
        choices_batch = batch["choices"]
        for q, choices in zip(questions, choices_batch):
            choices_text = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
            prompt = (
                f"User: Question: {q.strip()}\n"
                f"Choices:\n{choices_text}\n"
                "Answer only with the letter (A–E) corresponding to the correct choice.\n"
                "Assistant:"
            )
            prompts.append(prompt)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'DA' or 'MC'.")

    # --- Encode multimodal inputs ---
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        inputs_list = []
        for img, prompt in zip(images, prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            inputs_list.append(inp)
        return inputs_list
    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    gen_ids = model.generate(
        **inputs,
        eos_token_id=eos_token_id,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = postprocess_aokvqa_outputs(decoded)

    # --- For multiple-choice, map A–E letter → actual text ---
    if mode.upper() == "MC":
        mapped_results = []
        for r, choices in zip(results, batch["choices"]):
            letter = None
            for c in r:
                if c.upper() in ["A", "B", "C", "D", "E"]:
                    letter = c.upper()
                    break
            if letter:
                idx = ord(letter) - 65
                mapped_results.append(choices[idx] if idx < len(choices) else "unanswerable")
            else:
                mapped_results.append("unanswerable")
        results = mapped_results

    return results


@torch.no_grad()
def infer_idefics_sqa(
    model,
    processor,
    batch,
    max_new_tokens=80,
    return_inputs=False,
):
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

    def postprocess_scienceqa_output(texts):
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'")
            t = t.lower().replace("(a)", "a").replace("(b)", "b").replace("(c)", "c").replace("(d)", "d").replace("(e)", "e")

            if len(t) == 0:
                clean.append("E")  # default fallback
                continue
            first = t[0].upper()
            if first in ["A", "B", "C", "D", "E"]:
                clean.append(first)
            else:
                clean.append(t.split()[0].capitalize())
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    choices = batch["choices"]
    contexts = batch.get("contexts", None)
    lectures = batch.get("lectures", None)
    drafts = batch.get("draft", None)

    full_prompts = []
    for i, q in enumerate(questions):
        prompt = "User:\n"

        if lectures is not None and len(lectures) > i and lectures[i]:
            prompt += f"Lecture: {lectures[i].strip()}\n"
        if contexts is not None and len(contexts) > i and contexts[i]:
            prompt += f"Context: {contexts[i].strip()}\n"

        prompt += f"Question: {q.strip()}\n"

        if choices is not None and len(choices) > i:
            opts = choices[i]
            opts_str = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(opts)])
            prompt += f"Choices:\n{opts_str}\n"

        if drafts is not None and len(drafts) > i and drafts[i]:
            prompt += f"Initial answer was '{drafts[i]}'. Reconsider and correct if needed.\n"

        prompt += "Answer with ONLY the letter (A/B/C/D/E).<end_of_utterance>\nAssistant:"
        full_prompts.append(prompt)

    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        input_list = []
        for img, q in zip(images, full_prompts):
            inp = processor(
                text=q,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    generated_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    results = postprocess_scienceqa_output(decoded_texts)
    return results


@torch.no_grad()
def infer_idefics_mme(
    model,
    processor,
    batch,
    max_new_tokens=80,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for Idefics-9B-Instruct on MME Benchmark (Yes/No tasks).
    Generates strictly 'Yes' or 'No' answers based on visual evidence.
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

    def postprocess_mme_outputs(texts):
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'").lower()

            if "yes" in t and "no" not in t:
                clean.append("Yes")
                continue
            if "no" in t and "yes" not in t:
                clean.append("No")
                continue

            clean.append("")
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    tasks = batch.get("tasks", None)

    full_prompts = []
    for i, q in enumerate(questions):
        task_name = tasks[i] if tasks is not None and len(tasks) > i else "general"
        prompt = (
            f"User: Task: {task_name}\n"
            f"Question: {q.strip()}\n"
            "Answer strictly with 'Yes' or 'No' based on the image.<end_of_utterance>\nAssistant:"
        )
        full_prompts.append(prompt)

    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        input_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = postprocess_mme_outputs(decoded)
    return results


@torch.no_grad()
def infer_idefics_mmbench(
    model,
    processor,
    batch,
    max_new_tokens=80,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for Idefics-9B-Instruct on MMBench (Multiple Choice).
    Forces model to choose one of A/B/C/D explicitly and returns formatted output like 'A: bike'.
    """

    def _process_image(img):
        """Convert input to RGB PIL Image"""
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

    def postprocess_mmbench_outputs(texts, choices_list):
        """Extract choice letter (A/B/C/D) and map back to full text like 'A: bike'"""
        clean = []
        for t, choices in zip(texts, choices_list):
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'").upper()

            # detect A/B/C/D from text
            selected = None
            for c in ["A", "B", "C", "D"]:
                if t.startswith(c):
                    selected = c
                    break
                if f" {c}" in t:
                    selected = c
                    break

            # fallback by keyword match
            if selected is None:
                for c in ["A", "B", "C", "D"]:
                    if any(c.lower() in t.lower() for c in [f"({c})", f"{c}:", f"{c})"]):
                        selected = c
                        break

            if selected:
                full_choice = next((ch for ch in choices if ch.startswith(selected)), selected)
            else:
                full_choice = ""

            clean.append(full_choice)
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    choices_list = batch["choices"]
    tasks = batch.get("tasks", None)

    # --- Build prompts ---
    full_prompts = []
    for i, q in enumerate(questions):
        task_name = tasks[i] if tasks is not None and len(tasks) > i else "MMBench"
        choices_text = "\n".join(choices_list[i])
        prompt = (
            f"User: Task: {task_name}\n"
            f"Question: {q.strip()}\n"
            f"Choices:\n{choices_text}\n"
            "Answer with only the letter (A, B, C, or D) that best answers the question.<end_of_utterance>\nAssistant:"
        )
        full_prompts.append(prompt)

    # --- Encode inputs ---
    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        input_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    # --- Generate ---
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = postprocess_mmbench_outputs(decoded, choices_list)
    return results


@torch.no_grad()
def infer_idefics_seedbench(
    model,
    processor,
    batch,
    max_new_tokens=64,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for Idefics-9B-Instruct on SEED-Bench (multiple-choice image QA).
    - Generates single-letter answers (A/B/C/D).
    - Handles image-based questions only.
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

    def postprocess_seed_outputs(texts):
        """Extract single-letter (A/B/C/D) answers from outputs."""
        clean = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "").strip()
            t = t.strip(" .,:;!?\"'").upper()

            # Find single letter A/B/C/D
            if any(opt in t for opt in ["A", "B", "C", "D"]):
                for opt in ["A", "B", "C", "D"]:
                    if opt in t:
                        clean.append(opt)
                        break
            else:
                clean.append("")  # fallback for unclear outputs
        return clean

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    choices_list = batch["choices"]
    qtypes = batch.get("question_types", None)

    # Build prompts
    full_prompts = []
    for i, q in enumerate(questions):
        choices = choices_list[i]
        choice_text = "\n".join(choices)
        qtype = qtypes[i] if qtypes and len(qtypes) > i else "general"

        prompt = (
            f"User: Task: {qtype}\n"
            f"Question: {q.strip()}\n"
            f"{choice_text}\n"
            "Select the correct answer (A, B, C, or D) based on the image."
            "<end_of_utterance>\nAssistant:"
        )
        full_prompts.append(prompt)

    # Tokenize + preprocess
    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        input_list = []
        for img, prompt in zip(images, full_prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    # Generate answers
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = postprocess_seed_outputs(decoded)
    return results


@torch.no_grad()
def infer_idefics_haloquest(
    model,
    processor,
    batch,
    max_new_tokens=80,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for Idefics-9B-Instruct on HaloQuest dataset.
    - Produces concise, factual answers strictly based on the image.
    - Explicitly handles visibility, counting, and text-copying rules.
    - Removes conversational artifacts and returns clean outputs.
    """

    def _process_image(img):
        """Convert input into RGB PIL Image."""
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

    def _clean_outputs(texts):
        """Remove conversational artifacts and trailing punctuation."""
        cleaned = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "")
            t = t.strip(" \n\t.,:;!?\"'")
            cleaned.append(t)
        return cleaned

    # --- Extract fields ---
    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]

    # --- Build prompts ---
    prompts = []
    for q in questions:
        prompt = (
            "User: You are answering visual questions.\n"
            f"Question: {q.strip()}\n"
            "Answer based only on what is visible in the image.\n"
            "- If something is not visible, answer 'Not visible'.\n"
            "- For numbers, count exactly what you see.\n"
            "- For text, copy exactly the letters you can read.\n"
            "Give one short factual sentence.\n"
            "<end_of_utterance>\nAssistant:"
        )
        prompts.append(prompt)

    # --- Encode inputs ---
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if return_inputs:
        input_list = []
        for img, prompt in zip(images, prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    # --- Exit and bad tokens ---
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    # --- Logits mode ---
    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    # --- Generate ---
    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = _clean_outputs(decoded)
    return results


@torch.no_grad()
def infer_idefics_mmhalbench(
    model,
    processor,
    batch,
    max_new_tokens=80,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for Idefics-9B-Instruct on MMHal-Bench dataset.
    - Generates concise, factual, and grounded answers based on the image.
    - Handles multiple question types (e.g., attribute, counting, reasoning).
    - Cleans conversational artifacts and normalizes outputs.
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

    def _clean_outputs(texts):
        cleaned = []
        for t in texts:
            if "Assistant:" in t:
                t = t.split("Assistant:")[-1]
            t = t.replace("<end_of_utterance>", "")
            t = t.strip(" \n\t.,:;!?\"'")
            cleaned.append(t)
        return cleaned

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    q_types = batch.get("question_types", None)
    q_topics = batch.get("question_topics", None)

    prompts = []
    for i, q in enumerate(questions):
        qtype = q_types[i] if q_types is not None else "general"
        qtopic = q_topics[i] if q_topics is not None else ""
        q = q.strip()
        if not q.endswith("?"):
            q += "?"

        if qtype.lower() in ["yes/no", "binary"]:
            inst = "Answer strictly with 'Yes' or 'No' based on what is visible in the image."
        elif qtype.lower() in ["counting", "number"]:
            inst = "Count precisely what you can see in the image and answer with a number."
        elif qtype.lower() in ["text", "ocr"]:
            inst = "Read and copy exactly the text visible in the image."
        elif qtype.lower() in ["reasoning", "commonsense"]:
            inst = "Answer concisely using only the information visible in the image. Do not guess."
        else:
            inst = "Answer concisely and factually based only on the image."

        prompt = (
            f"User: You are answering visual questions about images.\n"
            f"Question Type: {qtype}\n"
            f"Topic: {qtopic}\n"
            f"Question: {q}\n"
            f"{inst}\n"
            "If the answer cannot be determined from the image, reply 'Not visible'.\n"
            "<end_of_utterance>\nAssistant:"
        )
        prompts.append(prompt)

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        add_end_of_utterance_token=False,
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    if return_inputs:
        input_list = []
        for img, prompt in zip(images, prompts):
            inp = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                add_end_of_utterance_token=False,
            )
            inp = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            input_list.append(inp)
        return input_list

    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
    results = _clean_outputs(decoded)

    return results
