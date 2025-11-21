import torch
import torch.nn.functional as F
import difflib
import re
import json
from PIL import Image


@torch.no_grad()
def infer_paligemma(
        model,
        processor,
        batch,
        max_new_tokens=200,
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

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        q_text = q
        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx is not None:
                q_text += f"\nExpert context: {ctx}"
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
        ).to(model.device)

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
            decoded = processor.decode(outputs[0], skip_special_tokens=True)
            decoded = decoded.split('\n')[-1].strip()
            results.append(decoded)
    return results

    
@torch.no_grad()
def infer_paligemma_vizwiz(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on VizWiz dataset.
    - Tailored prompt for visually-impaired user queries.
    - Encourages short answers or 'unanswerable' when unclear.
    - Compatible with Google PaliGemma model API.
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

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        # ① VizWiz style instruction
        q_text = (
            "You are assisting a visually impaired person who cannot see the image.\n"
            f"Question: {q.strip()}\n"
            "If the question cannot be answered from the image, reply with 'unanswerable'.\n"
            "Otherwise, answer briefly and clearly using only a few words."
        )

        # ② Optional expert context
        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nAdditional hints from expert models: {ctx.strip()}"

        # ③ Draft correction
        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"\nPrevious tentative answer: '{draft}'. "
                    "Revise if needed based on the visual evidence."
                )

        # ④ Process input (PaliGemma processor handles both image + text)
        inputs = processor(
            images=proc_img,
            text=q_text,
            return_tensors="pt"
        ).to(model.device)

        # ⑤ Inference logic
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
            decoded = processor.decode(outputs[0], skip_special_tokens=True)

            # ⑥ Post-processing
            decoded = decoded.split("\n")[-1].strip()
            # Remove any verbose answers (common in PaliGemma)
            for prefix in ["Answer:", "A:", "Response:"]:
                if decoded.lower().startswith(prefix.lower()):
                    decoded = decoded[len(prefix):].strip()
            results.append(decoded)
    return results


@torch.no_grad()
def infer_paligemma_gqa(
        model,
        processor,
        batch,
        max_new_tokens=100,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference function for PaliGemma on GQA dataset.
    - Focused on visual reasoning, attribute recognition, and relational understanding.
    - Answers are expected to be short (1–3 words), factual, and visually grounded.
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

        # --- Prompt Design (optimized for GQA) ---
        prompt = (
            f"Question: {q.strip()}\n"
            "Look at the image carefully and answer concisely (1–3 words).\n"
            "Focus on logical reasoning and visible evidence."
        )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx is not None and len(ctx.strip()) > 0:
                prompt += f"\nAdditional visual context: {ctx.strip()}"

        if draft_answers is not None and len(draft_answers) > i:
            d = draft_answers[i]
            if d is not None and len(d.strip()) > 0:
                prompt += f"\nPrevious draft answer: '{d.strip()}'. Correct it if wrong."

        # --- Tokenization and generation ---
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

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
            decoded = processor.decode(outputs[0], skip_special_tokens=True)
            decoded = decoded.split("\n")[-1].strip()
            results.append(decoded)

    return results


@torch.no_grad()
def infer_paligemma_textvqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on TextVQA (scene-text question answering).
    - Reads and interprets visible text in real-world images.
    - Supports optional OCR text and OCR tokens as context.
    - Encourages concise factual answers (1–5 words).
    - Replies 'unanswerable' when text is unclear, missing, or not relevant.
    - Compatible with Google PaliGemma model API.
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
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        return img

    images = batch["images"]
    questions = batch["questions"]
    ocr_texts = batch.get("ocr_text", None)
    ocr_tokens = batch.get("ocr_tokens", None) 
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        q_text = (
            "You are answering a question that requires reading text in the image.\n"
            "Carefully read all visible words, signs, and labels.\n"
            "If the answer cannot be determined from the text, reply exactly with 'unanswerable'.\n"
        )

        context_str = ""
        if ocr_texts is not None and len(ocr_texts) > i and ocr_texts[i]:
            context_str += f"Recognized text in the image: {ocr_texts[i].strip()}"
        if ocr_tokens is not None and len(ocr_tokens) > i and ocr_tokens[i]:
            token_str = " | ".join(ocr_tokens[i])
            context_str += ("\n" if context_str else "") + f"Detected tokens: {token_str}"

        if context_str:
            q_text += f"\n{context_str}"

        if draft_answers is not None and len(draft_answers) > i and draft_answers[i]:
            draft = draft_answers[i]
            q_text += (
                f"\nPrevious answer attempt: '{draft}'. "
                "Revise if it contradicts the visible text."
            )
            
        q_text += (
            f"\nQuestion: {q.strip()}"
            "\nProvide a concise factual answer (1–5 words)."
        )

        inputs = processor(
            images=proc_img,
            text=q_text,
            return_tensors="pt"
        ).to(model.device)

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
        decoded = processor.decode(outputs[0], skip_special_tokens=True)

        decoded = decoded.split("\n")[-1].strip()
        for prefix in ["Answer:", "A:", "Response:"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        decoded_lower = decoded.lower()
        bad_tokens = ["", "none", "unknown", "no idea", "can't tell", "nothing", "error"]
        if decoded_lower in bad_tokens or len(decoded) < 2:
            decoded = "unanswerable"

        if len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_paligemma_ocrvqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on OCR-VQA (book-cover reading).
    - Encourages concise factual answers (title, author, genre, etc.)
    - Replies 'unanswerable' if information cannot be found on the cover.
    - Compatible with Google PaliGemma model API.
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
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        return img

    images = batch["images"]
    questions = batch["questions"]
    ocr_tokens = batch.get("ocr_tokens", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        q_text = (
            "You are an expert at reading and interpreting book covers.\n"
            "Read the visible text on the cover carefully.\n"
            "If the answer cannot be determined from the text, reply exactly with 'unanswerable'.\n"
        )

        if ocr_tokens is not None and len(ocr_tokens) > i:
            tokens = ocr_tokens[i]
            if tokens:
                joined = " ".join(tokens[:50])
                q_text += f"\nRecognized text on the cover: {joined}"

        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            if draft:
                q_text += (
                    f"\nPrevious guess: '{draft}'. "
                    "Revise if it contradicts the text on the cover."
                )
                
        q_text += (
            f"\nQuestion: {q.strip()}"
            "\nProvide a concise factual answer (1–5 words)."
        )
        

        inputs = processor(
            images=proc_img,
            text=q_text,
            return_tensors="pt"
        ).to(model.device)

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
        decoded = processor.decode(outputs[0], skip_special_tokens=True)

        decoded = decoded.split("\n")[-1].strip()
        for prefix in ["Answer:", "A:", "Response:"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        decoded_lower = decoded.lower()
        bad_tokens = ["", "none", "unknown", "no idea", "can't tell", "nothing", "error"]
        if decoded_lower in bad_tokens or len(decoded) < 2:
            decoded = "unanswerable"

        if len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_paligemma_cococaption(
    model,
    processor,
    batch,
    max_new_tokens=50,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for PaliGemma (fine-tuned on COCO Captions).
    - Uses 'caption' prompt (as trained in PaliGemma-3B-ft-cococap-448).
    - Cleans prompt repetition from outputs.
    - Handles both single and batched inputs.
    """

    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        return img

    # Extract data from batch
    images = batch.get("images", [])
    if isinstance(images, (str, torch.Tensor, Image.Image)):
        images = [images]
    captions = batch.get("captions", None)

    results = []
    prompt = "caption en"

    for i, img in enumerate(images):
        proc_img = _process_image(img)

        inputs = processor(
            text=prompt,
            images=proc_img,
            return_tensors="pt"
        ).to(model.device)

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
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()

        if decoded.lower().startswith(prompt.lower()):
            decoded = decoded[len(prompt):].strip()
        for prefix in ["caption:", "description:", "the image shows", "image of", "this image shows"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        decoded = decoded.lstrip(":,.- ").strip()
        results.append(decoded)

    return results


@torch.no_grad()
def infer_paligemma_nocaps(
    model,
    processor,
    batch,
    max_new_tokens=60,
    temperature=0.7,
    top_p=0.9,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for PaliGemma on NoCaps dataset.
    - Uses a more descriptive prompt ("describe this image in detail").
    - Applies mild sampling (temperature, top_p) to handle open-vocabulary captions.
    - Cleans repetitive prefixes from output.
    """

    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        return img

    # Extract image(s)
    images = batch.get("images", [])
    if isinstance(images, (str, torch.Tensor, Image.Image)):
        images = [images]

    results = []
    # NoCaps requires more open-ended captioning
    prompt = "caption en"

    for img in images:
        proc_img = _process_image(img)
        inputs = processor(
            text=prompt,
            images=proc_img,
            return_tensors="pt"
        ).to(model.device)

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
            temperature=temperature,
            top_p=top_p,
            do_sample=True, 
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()

        for prefix in [prompt, "caption:", "description:", "the image shows", "image of", "this image shows"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        decoded = decoded.lstrip(":,.- ").strip()
        results.append(decoded)

    return results


@torch.no_grad()
def infer_paligemma_flickr(
    model,
    processor,
    batch,
    max_new_tokens=50,
    temperature=0.6,
    top_p=0.9,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for PaliGemma on Flickr30k captioning dataset.
    - Uses a moderate descriptive prompt ("describe this image briefly").
    - Balances between factual and expressive captions.
    - Cleans repeated prefixes or prompt residue.
    """

    def _process_image(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            if img.dtype.is_floating_point:
                img = (img.clamp(0, 1) * 255).byte()
            img = Image.fromarray(img.cpu().numpy()).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        return img

    images = batch.get("images", [])
    if isinstance(images, (str, torch.Tensor, Image.Image)):
        images = [images]

    results = []
    prompt = "describe this photo briefly in one sentence"

    for img in images:
        proc_img = _process_image(img)
        inputs = processor(
            text=prompt,
            images=proc_img,
            return_tensors="pt"
        ).to(model.device)

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
            temperature=temperature,
            top_p=top_p,
            do_sample=True,  # allow some diversity
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()

        # Remove unwanted prefixes or prompt residue
        for prefix in [prompt, "caption:", "description:", "the image shows", "image of", "this image shows"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        decoded = decoded.lstrip(":,.- ").strip()
        results.append(decoded)

    return results


@torch.no_grad()
def infer_paligemma_vcr(
    model,
    processor,
    batch,
    max_new_tokens=80,
    temperature=0.2,
    return_inputs=False,
    return_logits=False,
):
    """
    PaliGemma inference for VCR (QA→R) with ViCor-style unified reasoning.
    - Role: visual reasoning assistant
    - Output format: JSON with keys "answer_choice", "rationale_choice", "reasoning"
    - Ask for concise reasoning tied to image.
    """

    def _find_most_similar(text, choices):
        text = text.lower().strip()
        scores = [difflib.SequenceMatcher(None, text, c.lower()).ratio() for c in choices]
        return int(torch.tensor(scores).argmax().item())

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

    preds = []

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    answers = batch["answer_choices"]
    rationales = batch["rationales"]

    # ---------- Build improved prompt ----------
    full_prompts = []
    for i, q in enumerate(questions):
        ans_text = "\n".join([f"({j+1}) {a}" for j, a in enumerate(answers[i])])
        rat_text = "\n".join([f"({j+1}) {r}" for j, r in enumerate(rationales[i])])

        prompt = (
            "You are a visual reasoning assistant. Examine the image carefully and provide your answer and its rationale.\n"
            "    Step 1: Choose the correct answer choice number.\n"
            "    Step 2: Then choose the rationale choice number that best supports that answer.\n"
            "    Step 3: Write one **concise** sentence explaining **why** the selected answer is correct, referring to a specific visual clue in the image.\n"
            "\n"
            "Question: " + q.strip() + "\n"
            "Answer Choices:\n" + ans_text + "\n"
            "Rationale Choices:\n" + rat_text + "\n"
            "Important: Use exactly the following JSON format (and nothing else):\n"
            "{ \"answer_choice\": X, \"rationale_choice\": Y, \"reasoning\": \"<your one-sentence reasoning>\" }\n"
        )
        full_prompts.append(prompt)

    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    if return_inputs:
        input_list = []
        for i in range(len(images)):
            single_input = processor(
                text=full_prompts[i],
                images=images[i],
                return_tensors="pt"
            ).to(model.device)
            input_list.append(single_input)
        return input_list

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)

    for i, t in enumerate(decoded):
        t = t.strip()
        try:
            parsed = json.loads(t)
            ans_idx = parsed["answer_choice"] - 1
            rat_idx = parsed["rationale_choice"] - 1
        except Exception:
            ans_match = re.search(r"answer_choice\"\s*:\s*(\d+)", t, re.IGNORECASE)
            rat_match = re.search(r"rationale_choice\"\s*:\s*(\d+)", t, re.IGNORECASE)
            if ans_match:
                ans_idx = int(ans_match.group(1)) - 1
            else:
                ans_idx = _find_most_similar(t.split("reasoning")[0], answers[i])
            if rat_match:
                rat_idx = int(rat_match.group(1)) - 1
            else:
                rat_idx = _find_most_similar(t.split("reasoning")[-1], rationales[i])

        ans_text = answers[i][ans_idx] if 0 <= ans_idx < len(answers[i]) else "N/A"
        rat_text = rationales[i][rat_idx] if 0 <= rat_idx < len(rationales[i]) else "N/A"

        preds.append(f"Answer {ans_idx + 1}: {ans_text} | Rationale {rat_idx + 1}: {rat_text}")

    return preds


@torch.no_grad()
def infer_paligemma_vsr(
        model,
        processor,
        batch,
        max_new_tokens=10,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on the Visual Spatial Reasoning (VSR) dataset.
    - Reformulates samples as binary (Yes/No) spatial reasoning questions.
    - Uses deterministic generation to avoid ambiguous outputs.
    """

    def _process_image(img):
        """Convert image input (path or tensor) to RGB PIL image."""
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
    results = []

    for img, q in zip(images, questions):
        proc_img = _process_image(img)

        q_text = (
            f"{q}\n"
        )

        inputs = processor(
            images=proc_img,
            text=q_text,
            return_tensors="pt"
        ).to(model.device)

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
                temperature=0.0,
            )
            decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()
            if decoded.lower().endswith("yes"):
                pred = "yes"
            elif decoded.lower().endswith("no"):
                pred = "no"
            else:
                pred = decoded
            results.append(pred)
    return results


@torch.no_grad()
def infer_paligemma_okvqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on OK-VQA dataset.
    - Requires external / commonsense knowledge beyond image content.
    - Encourages concise factual answers (1–5 words).
    - Optionally supports expert context and draft refinement.
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

        # --- Prompt Design (optimized for OK-VQA) ---
        prompt = (
            "You are an expert visual reasoning assistant that can combine image understanding "
            "with world knowledge and commonsense reasoning.\n"
            f"Question: {q.strip()}\n"
            "Use both the visual scene and external knowledge to answer concisely (1–5 words)."
        )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx is not None and len(ctx.strip()) > 0:
                prompt += f"\nHelpful background knowledge: {ctx.strip()}"

        if draft_answers is not None and len(draft_answers) > i:
            d = draft_answers[i]
            if d is not None and len(d.strip()) > 0:
                prompt += f"\nPrevious tentative answer: '{d.strip()}'. Correct or refine it if necessary."

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

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
            decoded = processor.decode(outputs[0], skip_special_tokens=True)

            decoded = decoded.split("\n")[-1].strip()
            for prefix in ["Answer:", "A:", "Response:", "The answer is"]:
                if decoded.lower().startswith(prefix.lower()):
                    decoded = decoded[len(prefix):].strip()

            results.append(decoded)

    return results


@torch.no_grad()
def infer_paligemma_aokvqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on A-OKVQA (Augmented OK-VQA).
    Supports both Direct-Answer (DA) and Multiple-Choice (MC) modes,
    with explicit option text and post-processing of letter answers.
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
    answers_gt = batch.get("answers", None)
    choices = batch.get("choices", None)
    expert_context = batch.get("context", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        prompt = (
            "You are a knowledgeable multimodal reasoning assistant. "
            "Combine visual understanding with commonsense and world knowledge.\n"
            f"Question: {q.strip()}"
        )

        choice_labels = []
        if choices is not None and len(choices) > i:
            opts = choices[i]
            if opts not in [None, "", []]:
                choice_labels = [chr(65 + j) for j in range(len(opts))]  # ["A","B","C","D"]
                opt_text = "\n".join([f"({label}) {text}" for label, text in zip(choice_labels, opts)])
                prompt += f"\nOptions:\n{opt_text}\nChoose the best answer and respond with its content (not the letter)."

        if answers_gt is not None and len(answers_gt) > i and len(answers_gt[i]) > 0:
            uniq = sorted(set([a.strip().lower() for a in answers_gt[i]]))
            if len(uniq) > 0:
                prompt += f"\nAnnotator answers include: {', '.join(uniq[:3])}."

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx and len(ctx.strip()) > 0:
                prompt += f"\nHelpful background knowledge: {ctx.strip()}"
        if draft_answers is not None and len(draft_answers) > i:
            d = draft_answers[i]
            if d and len(d.strip()) > 0:
                prompt += f"\nPrevious tentative answer: '{d.strip()}'. Refine or correct if necessary."

        prompt += "\nAnswer concisely (1–5 words)."

        # --- ⑤ Encode (image + text) ---
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        # --- ⑥ Forward / generate ---
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

        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()
        decoded = decoded.split("\n")[-1].strip()
        for prefix in ["Answer:", "A:", "Response:", "The answer is"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        if len(decoded) == 1 and decoded.upper() in choice_labels:
            idx = ord(decoded.upper()) - 65
            if 0 <= idx < len(choices[i]):
                decoded = choices[i][idx]

        results.append(decoded)

    return results


@torch.no_grad()
def infer_paligemma_sqa(
    model,
    processor,
    batch,
    device="cuda:0",
    max_new_tokens=100,
    temperature=0.0,
    return_inputs=False,
    return_logits=False,
):
    """
    Args:
        model: PaliGemma model
        processor: corresponding processor (e.g., AutoProcessor.from_pretrained)
        batch: dict containing
            - images
            - questions
            - choices
            - lectures (optional)
            - contexts (optional)
        device: device string
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
    choices = batch["choices"]
    lectures = batch.get("lectures", None)
    contexts = batch.get("contexts", None)

    for i in range(len(questions)):
        img = _process_image(images[i])

        q = questions[i]
        chs = choices[i]
        ch_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(chs)])

        prompt_parts = []

        if lectures and len(lectures) > i and lectures[i]:
            prompt_parts.append(f"Lecture:\n{lectures[i].strip()}")
        if contexts and len(contexts) > i and contexts[i]:
            prompt_parts.append(f"Context:\n{contexts[i].strip()}")
        prompt_parts.append(f"Question: {q.strip()}\nChoices:\n{ch_text}")
        prompt_parts.append("Answer with the correct option letter (A, B, C, D, or E).")
        prompt = "\n\n".join(prompt_parts)

        inputs = processor(
            images=img,
            text=prompt,
            return_tensors="pt"
        ).to(device)

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            decoded = processor.decode(outputs[0], skip_special_tokens=True)
            decoded = decoded.strip().split("\n")[-1]
            results.append(decoded)

    return results


@torch.no_grad()
def infer_paligemma_mme(
        model,
        processor,
        batch,
        max_new_tokens=100,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on MME Benchmark (clean postprocessing version).
    Keeps only the final short answer (e.g., 'Yes' or 'No').
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

    perception_tasks = {
        "artwork", "celebrity", "color", "count", "existence",
        "landmark", "OCR", "position", "posters", "scene"
    }
    cognition_tasks = {
        "commonsense_reasoning", "code_reasoning",
        "numerical_calculation", "text_translation"
    }

    images = batch["images"]
    questions = batch["questions"]
    tasks = batch.get("tasks", ["unknown"] * len(images))
    results = []

    for img, q, t in zip(images, questions, tasks):
        proc_img = _process_image(img)
        t = t.lower()

        if t in perception_tasks:
            prompt = (
                f"Task: {t.replace('_', ' ')}.\n"
                f"Question: {q.strip()}\n"
                "Look carefully at the image and answer accurately.\n"
                "If it is a yes/no question, respond with 'Yes' or 'No'.\n"
                "If it requires a word or phrase, keep the answer concise (1–3 words)."
            )
        elif t in cognition_tasks:
            prompt = (
                f"Task: {t.replace('_', ' ')}.\n"
                f"Question: {q.strip()}\n"
                "Analyze and reason carefully using the visible content.\n"
                "Provide a short factual answer (1–5 words)."
            )
        else:
            prompt = (
                f"Question: {q.strip()}\n"
                "Provide a short, direct, and factual answer."
            )

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

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()

        lines = [l.strip() for l in decoded.splitlines() if l.strip()]
        final = lines[-1] if lines else decoded

        for prefix in ["Answer:", "A:", "Response:", "The answer is"]:
            if final.lower().startswith(prefix.lower()):
                final = final[len(prefix):].strip()

        if final.lower() in ["yes", "no"]:
            final = final.capitalize()

        results.append(final)

    return results


@torch.no_grad()
def infer_paligemma_mmbench(
        model,
        processor,
        batch,
        max_new_tokens=50,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on MMBench (Legacy) Dataset.
    - Prompt enforces model to answer with only A/B/C/D.
    - Returns formatted answers like 'A: bike'.
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

    # --- Load from batch ---
    images = batch["images"]
    questions = batch["questions"]
    choices_list = batch.get("choices", [[]] * len(images))
    hints = batch.get("hints", [""] * len(images))
    comments = batch.get("comments", [""] * len(images))
    tasks = batch.get("tasks", ["MMBench"] * len(images))

    results = []

    for img, q, choices, hint, comment, task in zip(images, questions, choices_list, hints, comments, tasks):
        proc_img = _process_image(img)

        choice_text = "\n".join(choices) if choices else ""
        hint_text = f"\nHint: {hint.strip()}" if hint else ""
        comment_text = f"\nComment: {comment.strip()}" if comment else ""

        prompt = (
            f"[Task: {task}]\n"
            f"Question: {q.strip()}\n"
        )
        if choice_text:
            prompt += f"{choice_text}\n"
        prompt += (
            f"{hint_text}{comment_text}\n"
            "Choose the most appropriate answer strictly by replying with only one letter among (A, B, C, D).\n"
            "For example, reply exactly as 'A' or 'B' — do not include text or explanation."
        )

        # --- Tokenize inputs ---
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()

        lines = [l.strip() for l in decoded.splitlines() if l.strip()]
        final = lines[-1] if lines else decoded
        for prefix in ["Answer:", "A:", "Response:", "The answer is"]:
            if final.lower().startswith(prefix.lower()):
                final = final[len(prefix):].strip()
        final = final.strip()

        letter = None
        if re.match(r"^[A-Da-d]", final):
            letter = final[0].upper()

        if letter and choices:
            for ch in choices:
                if ch.strip().upper().startswith(letter):
                    content = ch.split(".", 1)[-1].strip() if "." in ch else ch[1:].strip()
                    final = f"{letter}: {content}"
                    break
        else:
            lower_final = final.lower()
            for ch in choices:
                m = re.match(r"^([A-Da-d])[\.\)]\s*(.*)", ch.strip())
                if m:
                    c_letter, c_text = m.group(1).upper(), m.group(2).strip().lower()
                    if lower_final in c_text or c_text in lower_final:
                        final = f"{c_letter}: {m.group(2).strip()}"
                        break

        results.append(final)

    return results


@torch.no_grad()
def infer_pali_seedbench(
        model,
        processor,
        batch,
        max_new_tokens=60,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on SEED-Bench (multiple-choice).
    - Supports image-based multiple-choice questions (A–D).
    - Returns concise predicted answers ("A", "B", "C", or "D").
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
    choices = batch["choices"]
    qtypes = batch.get("question_types", ["Unknown"] * len(images))

    results = []

    for img, q, opts, qtype in zip(images, questions, choices, qtypes):
        proc_img = _process_image(img)

        # Build prompt for multiple-choice reasoning
        prompt = (
            f"Task: {qtype}.\n"
            f"Question: {q.strip()}\n\n"
            "Choices:\n" + "\n".join(opts) + "\n\n"
            "Select the correct answer (A, B, C, or D) based on the image content."
        )

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # Generate prediction
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()

        lines = [l.strip() for l in decoded.splitlines() if l.strip()]
        final = lines[-1] if lines else decoded

        for prefix in ["Answer:", "A:", "Response:", "The answer is"]:
            if final.lower().startswith(prefix.lower()):
                final = final[len(prefix):].strip()

        import re
        match = re.search(r"\b([A-D])\b", final, re.IGNORECASE)
        if match:
            final = match.group(1).upper()
        else:
            final = final.strip()[:1].upper() if final else ""

        results.append(final)

    return results


@torch.no_grad()
def infer_pali_haloquest(
        model,
        processor,
        batch,
        max_new_tokens=80,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on HaloQuest dataset.
    - Type-agnostic but hallucination-resistant prompt.
    - Encourages visual grounding, uncertainty awareness, and factual reasoning.
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

    results = []
    images = batch["images"]
    questions = batch["questions"]

    for img, q in zip(images, questions):
        proc_img = _process_image(img)

        prompt = (
            f"Look at the image carefully and answer the question strictly based on visible evidence.\n"
            f"If the object or detail asked about is not visible, explicitly say 'not visible'.\n"
            f"If the image does not provide enough information to be sure, say 'uncertain'.\n\n"
            f"Question: {q.strip()}\nAnswer in one short sentence:"
        )

        # prepare model inputs
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()
        lines = [l.strip() for l in decoded.splitlines() if l.strip()]
        final = lines[-1] if lines else decoded

        # simple clean
        for prefix in ["Answer:", "Response:", "A:", "Output:", "The answer is"]:
            if final.lower().startswith(prefix.lower()):
                final = final[len(prefix):].strip()
        final = final.strip().capitalize()

        results.append(final)

    return results


@torch.no_grad()
def infer_pali_mmhalbench(
        model,
        processor,
        batch,
        max_new_tokens=60,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for PaliGemma on MMHal-Bench dataset (no qtype exposure).
    - Model only sees the question and image.
    - Prompts encourage grounded, concise, and factual answers.
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

    results = []

    for img, q in zip(images, questions):
        proc_img = _process_image(img)

        # unified prompt (no question_type information)
        prompt = (
            "Answer the following question based only on what is visible in the image. "
            "Respond factually and concisely. "
            "If the visual evidence is unclear or missing, say 'not visible' or 'uncertain'.\n\n"
            f"Question: {q.strip()}\nAnswer:"
        )

        # prepare model input
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True).strip()
        lines = [l.strip() for l in decoded.splitlines() if l.strip()]
        final = lines[-1] if lines else decoded

        # clean output
        for prefix in ["Answer:", "Response:", "A:", "Output:", "The answer is"]:
            if final.lower().startswith(prefix.lower()):
                final = final[len(prefix):].strip()
        final = final.replace(".", "").strip().capitalize()

        results.append(final)

    return results