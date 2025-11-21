import torch
import torch.nn.functional as F
import difflib
import random
from PIL import Image
import re


@torch.no_grad()
def infer_instructblip(
        model, 
        processor,
        batch,
        max_new_tokens=50,
        return_logits=False,
        return_inputs=False,
    ):
    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        return img

    images = batch['images']
    questions = batch['questions']
    expert_context = batch.get('context', None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        q_text = q
        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nContext: {ctx}"

        q_text += "\nAnswer the question using a single word or phrase."

        if "draft" in batch and len(batch["draft"]) > i:
            q_text += (
                f"\nThe initial answer was: '{batch['draft'][i]}'. "
                f"Now, given the new image evidence, please reconsider and provide a potentially corrected answer."
            )

        inputs = processor(
            images=proc_img,
            text=q_text,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
        elif return_inputs:
            results.append(inputs)
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            decoded = decoded.split('Answer the question using a single word or phrase.')[-1]
            decoded = decoded.split('please reconsider and provide a potentially corrected answer.')[-1]

            results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_instructblip_vizwiz(
        model, 
        processor,
        batch,
        max_new_tokens=30,
        return_logits=False,
        return_inputs=False,
    ):
    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        return img

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        # --- Prompt Design ---
        prompt = (
            f"Question: {q.strip()}\n"
            "Look at the image carefully and provide a short, direct answer (1–3 words).\n"
            "If you cannot clearly read, identify, or see the information, reply exactly with 'unanswerable'."
        )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                prompt += f"\nAdditional context: {ctx.strip()}"

        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            prompt += (
                f"\nPrevious answer was '{draft}'. "
                "If the visible evidence contradicts it, revise the answer accordingly."
            )

        prompt += "\nAnswer:"

        # --- Preprocess ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        elif return_inputs:
            results.append(inputs)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # --- Post-processing ---
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:")[-1]

        decoded = decoded.strip()
        decoded_lower = decoded.lower()

        # meaningless or blank → unanswerable
        bad_tokens = ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]
        if decoded_lower in bad_tokens:
            decoded = "unanswerable"

        # overly generic yes/no filtering (for ambiguous visual questions)
        if any(kw in q.lower() for kw in [
            "can you", "is there", "does it", "do you", "are there",
            "is this", "are you able", "would you", "could you"
        ]) and decoded_lower in ["yes", "no"]:
            decoded = "unanswerable"

        # question-type filter for open queries
        if any(kw in q.lower() for kw in ["what", "where", "which", "how", "tell", "describe"]) \
           and decoded_lower in ["yes", "no"]:
            decoded = "unanswerable"

        # overly long → truncate
        if len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_instructblip_gqa(
        model, 
        processor,
        batch,
        max_new_tokens=40,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for InstructBLIP on GQA dataset.
    - Emphasizes reasoning and short, factual answers.
    - Automatically handles optional context or draft answers.
    """

    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        return img

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        # --- Prompt Design (GQA-tailored) ---
        
        prompt = (
            f"Question: {q.strip()}\n"
            "Look at the image carefully and answer concisely (1–3 words).\n"
            "Focus on logical reasoning and visual evidence only."
        )

        # optional context (object detections, scene tags, etc.)
        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                prompt += f"\nAdditional context: {ctx.strip()}"

        # optional previous draft
        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            prompt += (
                f"\nPrevious answer was '{draft}'. "
                "If it seems inconsistent with the image, revise accordingly."
            )

        prompt += "\nAnswer:"

        # --- Preprocess ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        elif return_inputs:
            results.append(inputs)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # --- Post-processing ---
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:")[-1]

        decoded = decoded.strip()
        decoded_lower = decoded.lower()

        # filter out invalid outputs
        bad_tokens = ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]
        if decoded_lower in bad_tokens:
            decoded = "unanswerable"

        # overly long → truncate
        if len(decoded.split()) > 5:
            decoded = " ".join(decoded.split()[:5])

        # normalize yes/no answers for consistency
        if decoded_lower in ["yeah", "yep", "affirmative", "correct"]:
            decoded = "yes"
        elif decoded_lower in ["nope", "negative", "incorrect"]:
            decoded = "no"

        results.append(decoded.strip())

    return results

    
@torch.no_grad()
def infer_instructblip_textvqa(
        model, 
        processor,
        batch,
        max_new_tokens=30,
        return_logits=False,
        return_inputs=False,
    ):
    """
    InstructBLIP inference for TextVQA (scene-text reasoning).
    - Designed to handle questions that require reading visible text in images.
    - Encourages concise factual answers (1–5 words).
    - Replies 'unanswerable' if the text is unclear, missing, or unreadable.
    """

    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        return img

    images = batch["images"]
    questions = batch["questions"]
    ocr_tokens = batch.get("ocr_tokens", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        # --- Prompt Design (TextVQA-oriented) ---
        prompt = (
            "Read all visible text in the image carefully (e.g., signs, labels, boards).\n"
            "Answer the question using only the text and context visible in the image.\n"
            "If the answer cannot be found or the text is unreadable, reply exactly with 'unanswerable'."
        )

        if ocr_tokens is not None and len(ocr_tokens) > i:
            tokens = ocr_tokens[i]
            if tokens:
                joined = " ".join(tokens[:50])
                prompt += f"\nRecognized text: {joined}"

        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            prompt += (
                f"\nPrevious answer was '{draft}'. "
                "If the text evidence contradicts it, provide a corrected answer."
            )

        prompt += (
            f"\nQuestion: {q.strip()}"
            "\nGive a concise factual answer (1–5 words)."
            )
        prompt += "\nAnswer:"

        # --- Preprocess ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        elif return_inputs:
            results.append(inputs)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # --- Post-processing ---
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:")[-1]

        decoded = decoded.strip()
        decoded_lower = decoded.lower()

        bad_tokens = ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]
        if decoded_lower in bad_tokens:
            decoded = "unanswerable"

        # Filter meaningless yes/no outputs for open-ended text reasoning
        if any(kw in q.lower() for kw in [
            "what", "where", "which", "how", "tell", "describe", "read", "say", "text", "word"
        ]) and decoded_lower in ["yes", "no"]:
            decoded = "unanswerable"

        # truncate long responses
        if len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_instructblip_ocrvqa(
        model,
        processor,
        batch,
        max_new_tokens=30,
        return_logits=False,
        return_inputs=False,
    ):
    """
    InstructBLIP inference for OCR-VQA (book-cover reading).
    Reads visible text to answer factual questions about title, author, genre, etc.
    """

    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        return img

    images = batch["images"]
    questions = batch["questions"]
    ocr_tokens = batch.get("ocr_tokens", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        # --- Prompt Design ---
        prompt = (
            # f"Question: {q.strip()}\n"
            "Read the text visible on the book cover carefully.\n"
            "If the answer cannot be found from the visible text, reply exactly with 'unanswerable'.\n"
        )

        # Add OCR tokens as optional context
        if ocr_tokens is not None and len(ocr_tokens) > i:
            tokens = ocr_tokens[i]
            if tokens:
                joined = " ".join(tokens[:50])
                prompt += f"\nRecognized text: {joined}"

        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            prompt += (
                f"\nPrevious answer was '{draft}'. "
                "Revise it if incorrect based on the visible text."
            )
        
        prompt += (
            f"\nQuestion: {q.strip()}"
            "\nGive a concise factual answer (1–5 words)."
        )

        prompt += "\nAnswer:"

        # --- Preprocess ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        elif return_inputs:
            results.append(inputs)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # --- Post-processing ---
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:")[-1]

        decoded = decoded.strip()
        decoded_lower = decoded.lower()

        # blank or meaningless → unanswerable
        bad_tokens = ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]
        if decoded_lower in bad_tokens:
            decoded = "unanswerable"

        # remove excess chatter
        for prefix in ["Response:", "Assistant:", "The answer is", "Answer"]:
            if decoded_lower.startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        # overly long → truncate
        if len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_instructblip_cococaption(
    model,
    processor,
    batch,
    max_new_tokens: int = 50,
    return_logits: bool = False,
    return_inputs: bool = False,
):
    """
    Captioning inference for InstructBLIP (image → text).
    - Automatically adds 'Describe the image.' prompt.
    - Supports batch with 'images' key (list of paths or tensors).
    - Returns either text captions, logits, or inputs.
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

    images = batch["images"]
    captions = []

    for img in images:
        proc_img = _process_image(img)

        # --- Captioning Prompt ---
        prompt = "caption en"

        # --- Preprocess ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            captions.append(outputs.logits)
            continue
        elif return_inputs:
            captions.append(inputs)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # --- Clean caption ---
        decoded = decoded.replace(prompt, "").strip()
        decoded = decoded.split("\n")[0].strip()
        captions.append(decoded)

    return captions


@torch.no_grad()
def infer_instructblip_nocaps(
    model,
    processor,
    batch,
    max_new_tokens: int = 40,
    num_beams: int = 5,
    return_logits: bool = False,
    return_inputs: bool = False,
):
    """
    Optimized inference for InstructBLIP on NoCaps.
    - Uses beam search decoding (no sampling).
    - Cleans verbose patterns for CIDEr/SPICE-friendly captions.
    - Supports returning logits or inputs for debugging or analysis.
    """

    def _process_image(img):
        """Convert input to RGB PIL Image."""
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

    def _clean_caption(text: str) -> str:
        """Normalize and shorten caption for evaluation."""
        text = text.lower().strip()
        text = re.sub(r"^(the (image|photo|picture|scene) (shows|depicts|contains)\s*)", "", text)
        text = re.sub(r"(?i)\b(in the (scene|image|photo|picture)|there (is|are))\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) > 15:
            text = " ".join(text.split()[:15])
        if text and not text.endswith("."):
            text += "."
        return text.capitalize()

    images = batch["images"]
    captions = []

    for img in images:
        proc_img = _process_image(img)
        prompt = random.choice(["caption", "caption en", "caption this image"])

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        # --- Optional: return logits or inputs ---
        if return_logits:
            outputs = model(**inputs)
            captions.append(outputs.logits)
            continue
        elif return_inputs:
            captions.append(inputs)
            continue

        # --- Beam search decoding ---
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=1,
            length_penalty=1.0,
            repetition_penalty=1.05,
            early_stopping=True,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded_list = processor.batch_decode(outputs, skip_special_tokens=True)
        decoded_list = [d.replace(prompt, "").strip().split("\n")[0].strip().strip('"') for d in decoded_list]
        decoded_list = [_clean_caption(d) for d in decoded_list if len(d) > 0]

        best_caption = decoded_list[0] if decoded_list else ""
        captions.append(best_caption)

    return captions


@torch.no_grad()
def infer_instructblip_flickr(
    model,
    processor,
    batch,
    max_new_tokens: int = 50,
    return_logits: bool = False,
    return_inputs: bool = False,
):
    """
    Captioning inference for InstructBLIP (image → text).
    - Automatically adds 'Describe the image.' prompt.
    - Supports batch with 'images' key (list of paths or tensors).
    - Returns either text captions, logits, or inputs.
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

    images = batch["images"]
    captions = []

    for img in images:
        proc_img = _process_image(img)

        # --- Captioning Prompt ---
        prompt = "caption en"

        # --- Preprocess ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            captions.append(outputs.logits)
            continue
        elif return_inputs:
            captions.append(inputs)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # --- Clean caption ---
        decoded = decoded.replace(prompt, "").strip()
        decoded = decoded.split("\n")[0].strip()
        captions.append(decoded)

    return captions


@torch.no_grad()
def infer_instructblip_vcr(
    model,
    processor,
    batch,
    max_new_tokens=40,
    batch_size=4,
    return_inputs=False,
    return_logits=False,
):
    """
    InstructBLIP VCR inference with *prompt-only* changes.
    - Keeps the original single-pass generate() pipeline.
    - No ranking, no logits post-scoring, no external examples.
    - Tight, deterministic instruction to stabilize indices.
    Returns [(ans_idx+1, rat_idx+1), ...].
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
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        return img

    def _clean(t):
        if "Answer:" in t:
            t = t.split("Answer:")[-1]
        if "Assistant:" in t:
            t = t.split("Assistant:")[-1]
        return t.replace("<end_of_utterance>", "").strip(" .,:;!?\"'\n\t")

    def _find_most_similar(text, choices):
        text = text.lower().strip()
        scores = [difflib.SequenceMatcher(None, text, c.lower()).ratio() for c in choices]
        return int(torch.tensor(scores).argmax().item())

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    answers = batch["answer_choices"]
    rationales = batch["rationales"]
    drafts = batch.get("draft", None)
    expert_context = batch.get("context", None)

    full_prompts = []
    for i, q in enumerate(questions):
        ans_text = "\n".join([f"({j+1}) {a}" for j, a in enumerate(answers[i])])
        rat_text = "\n".join([f"({j+1}) {r}" for j, r in enumerate(rationales[i])])

        ctx = ""
        if expert_context is not None and len(expert_context) > i and expert_context[i]:
            ctx = f"Context: {expert_context[i].strip()}\n"

        prompt = (
            f"{ctx}"
            "Task: Read the question, resolve any references like [personX], and carefully compare the image with each choice.\n"
            f"Question: {q.strip()}\n"
            f"Answer choices (indices are 1..4):\n{ans_text}\n"
            f"Rationale choices (indices are 1..4):\n{rat_text}\n"
            "Rules:\n"
            "- Output exactly one line.\n"
            "- First pick the best answer index in {1,2,3,4}.\n"
            "- Then pick the best rationale index in {1,2,3,4} that specifically supports that answer.\n"
            "- Do not copy or rewrite any choice text. Do not add extra words.\n"
            'Format: (X) because (Y)\n'
        )

        if drafts is not None and len(drafts) > i and drafts[i]:
            prompt += f"Previous attempt index pair (may be wrong): {drafts[i]}\n"

        prompt += "Answer:"
        full_prompts.append(prompt)

    preds = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch_imgs = images[start:end]
        batch_prompts = full_prompts[start:end]

        inputs = processor(
            images=batch_imgs,
            text=batch_prompts,
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(model.device, torch.float16)

        if return_inputs:
            input_list = []
            for img, prmpt in zip(batch_imgs, batch_prompts):
                single_input = processor(
                    images=img,
                    text=prmpt,
                    return_tensors="pt"
                ).to(model.device)
                input_list.append(single_input)
            preds.extend(input_list)
            continue

        if return_logits:
            outputs = model(**inputs)
            preds.extend([outputs.logits] * len(batch_imgs))
            continue

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=0,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)

        for i, text in enumerate(decoded):
            cleaned = _clean(text)

            m = re.search(r"\((\d)\)\s*because\s*\((\d)\)", cleaned, re.IGNORECASE)
            if m:
                a_idx = int(m.group(1)) - 1
                r_idx = int(m.group(2)) - 1
            else:
                am = re.search(r"answer[^0-9]*\((\d)\)", cleaned, re.IGNORECASE)
                rm = re.search(r"rationale[^0-9]*\((\d)\)", cleaned, re.IGNORECASE)
                if am and rm:
                    a_idx = int(am.group(1)) - 1
                    r_idx = int(rm.group(1)) - 1
                else:
                    parts = cleaned.lower().split("because")
                    a_idx = _find_most_similar(parts[0] if parts else cleaned, answers[start + i])
                    r_idx = _find_most_similar(parts[-1] if parts else cleaned, rationales[start + i])

            a_idx = max(0, min(3, a_idx))
            r_idx = max(0, min(3, r_idx))

            ans_text = answers[start + i][a_idx] if 0 <= a_idx < len(answers[start + i]) else "N/A"
            rat_text = rationales[start + i][r_idx] if 0 <= r_idx < len(rationales[start + i]) else "N/A"

            preds.append(f"Answer {a_idx + 1}: {ans_text} | Rationale {r_idx + 1}: {rat_text}")

    return preds


@torch.no_grad()
def infer_instructblip_vsr(
        model,
        processor,
        batch,
        max_new_tokens=10,
        return_logits=False,
        return_inputs=False,
    ):
    """
    InstructBLIP inference for Visual Spatial Reasoning (VSR)
    - Removes explicit 'Yes/No' from prompt to avoid linguistic bias.
    - Lets model infer naturally, then postprocess output to Yes/No.
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
        return img

    images = batch["images"]
    questions = batch["questions"]
    results = []

    for img, q in zip(images, questions):
        proc_img = _process_image(img)

        # No "Yes/No" bias in the instruction
        q_text = (
            f"Look at the image and determine if the described relation is correct:\n"
            f"{q.strip()}"
        )

        inputs = processor(
            images=proc_img,
            text=q_text,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
        elif return_inputs:
            results.append(inputs)
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip().lower()
            decoded = decoded.split('.')[-1].strip()
            if decoded.startswith("yes"):
                decoded = "yes"
            elif decoded.startswith("no"):
                decoded = "no"
            results.append(decoded)

    return results


@torch.no_grad()
def infer_instructblip_okvqa(
        model, 
        processor,
        batch,
        max_new_tokens=60,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for InstructBLIP on OK-VQA dataset.
    - Combines visual understanding with commonsense and world knowledge.
    - Encourages concise factual answers (1–5 words).
    - Compatible with expert context (retrieved knowledge, captions, etc.) and draft refinement.
    """

    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        return img

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        # --- Prompt Design (OK-VQA-specific) ---
        prompt = (
            f"Question: {q.strip()}\n"
            "Use your visual understanding and commonsense knowledge to answer accurately.\n"
            "The answer should be short (1–5 words) and based on both the image and external knowledge.\n"
            "Answer:"
        )

        # Optional context (e.g., retrieved wiki facts, object captions)
        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                prompt = (
                    f"Background knowledge: {ctx.strip()}\n" + prompt
                )

        # Optional previous draft refinement
        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            if draft:
                prompt = (
                    prompt
                    + f"\nPrevious answer: '{draft}'. "
                      "Revise it if it conflicts with visual or factual evidence."
                )

        # --- Preprocessing ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        elif return_inputs:
            results.append(inputs)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # --- Post-processing ---
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:")[-1]

        decoded = decoded.strip()
        decoded_lower = decoded.lower()

        # Invalid / meaningless responses
        bad_tokens = ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]
        if decoded_lower in bad_tokens:
            decoded = "unanswerable"

        # Overly long → truncate (5 words max)
        if len(decoded.split()) > 5:
            decoded = " ".join(decoded.split()[:5])

        # Normalize yes/no variants
        if decoded_lower in ["yeah", "yep", "affirmative", "correct"]:
            decoded = "yes"
        elif decoded_lower in ["nope", "negative", "incorrect"]:
            decoded = "no"

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_instructblip_aokvqa(
        model,
        processor,
        batch,
        max_new_tokens: int = 80,
        mode: str = "MC",  # "MC" (multiple-choice) or "DA" (direct-answer)
        return_logits: bool = False,
        return_inputs: bool = False,
    ):
    """
    Inference for InstructBLIP on A-OKVQA dataset.
    - Supports both Direct-Answer (DA) and Multiple-Choice (MC) formats.
    - In MC mode, returns the *text* corresponding to the predicted letter (A–E).
    - In DA mode, returns short factual answers (1–5 words).
    """

    def _process_image(img):
        """Convert tensor or path to RGB PIL image."""
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        return img

    images = batch["images"]
    questions = batch["questions"]
    choices = batch.get("choices", None)
    expert_context = batch.get("context", None)
    draft_answers = batch.get("draft", None)

    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        if mode == "MC" and choices is not None:
            opts = choices[i]
            opts_text = "\n".join([f"({chr(65+j)}) {opt}" for j, opt in enumerate(opts)])
            prompt = (
                f"Question: {q.strip()}\n"
                f"{opts_text}\n"
                "Answer with the letter (A–E) only, based on the image and commonsense knowledge.\n"
                "Answer:"
            )
        else:
            prompt = (
                f"Question: {q.strip()}\n"
                "Use both visual understanding and commonsense knowledge to answer concisely (1–5 words).\n"
                "Answer:"
            )

        # Optional retrieved knowledge or caption context
        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                prompt = f"Background knowledge: {ctx.strip()}\n" + prompt

        # Optional refinement with previous draft
        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            if draft:
                prompt += f"\nPrevious answer: '{draft}'. Revise it if incorrect."

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        elif return_inputs:
            results.append(inputs)
            continue

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        if "Answer:" in decoded:
            decoded = decoded.split("Answer:")[-1]
        decoded = decoded.strip()

        if mode == "MC":
            # Extract predicted letter and map to actual option text
            match = re.search(r"\b([A-Ea-e])\b", decoded)
            if match:
                letter = match.group(1).upper()
                idx = ord(letter) - 65
                if 0 <= idx < len(choices[i]):
                    decoded = choices[i][idx]
                else:
                    decoded = "unanswerable"
            else:
                decoded = "unanswerable"
        else:
            # Clean open-ended answer
            decoded_lower = decoded.lower()
            bad_tokens = ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]
            if decoded_lower in bad_tokens:
                decoded = "unanswerable"
            if len(decoded.split()) > 5:
                decoded = " ".join(decoded.split()[:5])
            if decoded_lower in ["yeah", "yep", "affirmative", "correct"]:
                decoded = "yes"
            elif decoded_lower in ["nope", "negative", "incorrect"]:
                decoded = "no"

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_instructblip_sqa(
    model,
    processor,
    batch,
    max_new_tokens=30,
    do_sample=False,
    top_p=0.9,
    top_k=5,
    return_logits=False,
    return_inputs=False,
):
    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        return img

    images = batch["images"]
    questions = batch["questions"]
    choices = batch.get("choices", None)
    lectures = batch.get("lecture", None)

    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        prompt = (
            "You are a science expert. "
            "Read the following carefully. "
            "Respond with exactly one uppercase letter (A, B, C, D, or E) — "
            "no explanations, no extra text.\n\n"
        )

        if lectures and len(lectures) > i and lectures[i]:
            prompt += f"Lecture:\n{lectures[i].strip()}\n\n"

        prompt += f"Question:\n{q.strip()}\n"
        if choices and len(choices) > i:
            c_list = choices[i]
            choice_lines = [f"{chr(65+j)}. {c}" for j, c in enumerate(c_list)]
            prompt += "Choices:\n" + "\n".join(choice_lines) + "\n"
        prompt += "\nAnswer (choose A, B, C, D, or E):"

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt",
        ).to(model.device, torch.float16)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        if return_inputs:
            results.append(inputs)
            continue

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        pattern = r"Answer \(choose[^)]*\):\s*([A-E])\b"
        match = re.search(pattern, decoded, re.IGNORECASE)

        if match:
            ans = match.group(1).upper()
        else:
            tail = decoded.strip().split()[-1].strip(".").upper()
            ans = tail if tail in {"A", "B", "C", "D", "E"} else "A"

        results.append(ans)

    return results


@torch.no_grad()
def infer_instructblip_mme(
        model,
        processor,
        batch,
        max_new_tokens=50,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for InstructBLIP on MME Benchmark (clean output version).
    Removes prompt repetition and enforces clean 'Yes'/'No' answers.
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

    images = batch["images"]
    questions = batch["questions"]
    tasks = batch.get("tasks", ["unknown"] * len(images))
    results = []

    for img, q, t in zip(images, questions, tasks):
        proc_img = _process_image(img)
        t = t.lower()

        # --- Simplify prompt (avoid "Please answer..." to prevent echo) ---
        if "?" not in q:
            q = q.strip() + "?"

        prompt = f"Answer this question with only one word: yes or no.\nQuestion: {q.strip()}"

        # --- Encode input ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        if return_inputs:
            results.append(inputs)
            continue

        # --- Generate ---
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # --- Clean extraneous repeats ---
        decoded = decoded.replace(prompt, "").strip()
        decoded = decoded.replace(q.strip(), "").strip()

        # --- Normalize ---
        decoded_lower = decoded.lower()
        if "yes" in decoded_lower and "no" not in decoded_lower:
            decoded = "Yes"
        elif "no" in decoded_lower and "yes" not in decoded_lower:
            decoded = "No"
        elif decoded_lower.strip() in ["yes", "no"]:
            decoded = decoded.capitalize()
        else:
            decoded = "Yes" if any(x in decoded_lower for x in ["yeah", "correct", "true"]) else \
                      "No" if any(x in decoded_lower for x in ["not", "false", "wrong"]) else ""

        results.append(decoded)

    return results


@torch.no_grad()
def infer_instructblip_mmbench(
        model,
        processor,
        batch,
        max_new_tokens=50,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for InstructBLIP on MMBench (multiple-choice tasks).
    - Each sample includes question + 4 choices (A, B, C, D)
    - Model is prompted to answer with only one of A/B/C/D
    - Returns clean 'A: text' format
    """

    def _process_image(img):
        """Ensure image is PIL RGB."""
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

    images = batch["images"]
    questions = batch["questions"]
    choices_list = batch.get("choices", [[] for _ in images])
    tasks = batch.get("tasks", ["unknown"] * len(images))

    results = []

    for img, q, choices, t in zip(images, questions, choices_list, tasks):
        proc_img = _process_image(img)
        t = t.lower()

        # --- Build multiple-choice prompt ---
        formatted_choices = "\n".join(choices)
        prompt = (
            f"{q.strip()}\n\n"
            f"Choices:\n{formatted_choices}\n\n"
            f"Answer with only the letter (A, B, C, or D)."
        )

        # --- Encode ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        if return_inputs:
            results.append(inputs)
            continue

        # --- Generate ---
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # --- Clean and extract letter ---
        decoded = decoded.replace(prompt, "").strip()
        decoded_lower = decoded.lower()

        # Normalize outputs
        letter = ""
        for opt in ["A", "B", "C", "D"]:
            if opt.lower() in decoded_lower:
                letter = opt
                break

        # Fallback heuristic (e.g., "Option A" or "choice B")
        if not letter:
            for opt in ["a", "b", "c", "d"]:
                if f"option {opt}" in decoded_lower or f"choice {opt}" in decoded_lower:
                    letter = opt.upper()
                    break

        # --- Match letter to actual choice text ---
        if letter and any(c.startswith(f"{letter}:") for c in choices):
            final = next(c for c in choices if c.startswith(f"{letter}:"))
        else:
            final = letter or ""

        results.append(final)

    return results


@torch.no_grad()
def infer_instructblip_seedbench(
        model,
        processor,
        batch,
        max_new_tokens=20,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for InstructBLIP on SEED-Bench (image-only tasks).
    - Generates single-letter answers: A/B/C/D.
    - Cleans repetitive or verbose generations.
    - Uses simple multiple-choice prompt template.
    """

    def _process_image(img):
        """Convert image input to RGB PIL image."""
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

    images = batch["images"]
    questions = batch["questions"]
    choices_list = batch["choices"]
    results = []

    for img, q, choices in zip(images, questions, choices_list):
        proc_img = _process_image(img)

        # --- Build concise multiple-choice prompt ---
        joined_choices = "\n".join(choices)
        prompt = (
            "Answer the following multiple-choice question by selecting only the letter (A, B, C, or D).\n"
            f"Question: {q.strip()}\n"
            f"{joined_choices}\n"
            "Answer:"
        )

        # --- Encode input ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        if return_inputs:
            results.append(inputs)
            continue

        # --- Generate ---
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # --- Clean text ---
        decoded = decoded.replace(prompt, "").strip()
        decoded = decoded.replace(q.strip(), "").strip()

        # --- Normalize to A/B/C/D ---
        decoded_upper = decoded.upper()
        if "A" in decoded_upper and all(x not in decoded_upper for x in ["B", "C", "D"]):
            decoded = "A"
        elif "B" in decoded_upper and all(x not in decoded_upper for x in ["A", "C", "D"]):
            decoded = "B"
        elif "C" in decoded_upper and all(x not in decoded_upper for x in ["A", "B", "D"]):
            decoded = "C"
        elif "D" in decoded_upper and all(x not in decoded_upper for x in ["A", "B", "C"]):
            decoded = "D"
        else:
            # fallback: first letter if clearly starts with A-D
            decoded = decoded_upper.strip()[:1] if decoded_upper[:1] in ["A", "B", "C", "D"] else ""

        results.append(decoded)

    return results


@torch.no_grad()
def infer_instructblip_haloquest(
    model,
    processor,
    batch,
    max_new_tokens=30,
    return_logits=False,
    return_inputs=False,
):
    """
    Reliable InstructBLIP inference for HaloQuest.
    - Simplified prompt to avoid echoing.
    - Ensures short, factual answers or fixed fallback words.
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

    images = batch["images"]
    questions = batch["questions"]
    results = []

    for img, q in zip(images, questions):
        proc_img = _process_image(img)

        prompt = (
            f"Question: {q}\n"
            "Answer the question based on the image.\n"
            "If not visible, reply 'Not visible'. "
            "If object does not exist, reply 'No such object'. "
            "If text unreadable, reply 'Unreadable'. "
            "Otherwise, reply with one short factual phrase.\n"
            "Answer:"
        )

        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded = decoded.replace(prompt, "").strip()

        decoded = re.sub(r"(?i)^(question|answer)\s*[:\-]?", "", decoded).strip()
        decoded = decoded.split("\n")[0].strip(",. ")

        if len(decoded.split()) > 8:
            decoded = " ".join(decoded.split()[:8])

        if decoded == "" or decoded.lower().startswith("look carefully"):
            decoded = "Not visible"

        results.append(decoded.capitalize())

    return results


@torch.no_grad()
def infer_instructblip_mmhalbench(
        model,
        processor,
        batch,
        max_new_tokens=60,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for InstructBLIP on MMHal-Bench.
    - Designed for hallucination evaluation.
    - Uses strict visual grounding with no access to ground-truth or topic metadata.
    - For ambiguous or invisible cases, must reply exactly 'uncertain'.
    """

    def _process_image(img):
        """Convert input (path/tensor/PIL) to RGB PIL."""
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
    images = batch["images"]
    questions = batch["questions"]

    for img, q in zip(images, questions):
        proc_img = _process_image(img)

        # --- Prompt enforcing grounded reasoning ---
        prompt = (
            f"Question: {q.strip()}\n"
            "Look carefully at the image and answer based only on visible evidence.\n"
            "Do not imagine or infer unseen details.\n"
            "If the answer cannot be determined from the image, reply exactly 'uncertain'.\n"
            "Answer:"
        )

        # --- Tokenization ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # --- Post-process ---
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:")[-1]
        decoded = decoded.strip().lower()

        # fallback normalization
        if decoded == "" or decoded in ["none", "unknown", "no idea", "can't tell", "not sure"]:
            decoded = "uncertain"
        if len(decoded.split()) > 5:
            decoded = " ".join(decoded.split()[:5])

        results.append(decoded)

    return results