import torch
from concurrent.futures import ThreadPoolExecutor
import re
import difflib
from PIL import Image


@torch.no_grad()
def infer_llava(
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

    images = batch['images']
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

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            images=proc_img, text=prompt, 
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
            decoded = decoded.split("[/INST]")[-1]
            results.append(decoded.strip())
    return results


@torch.no_grad()
def infer_llava_vizwiz(
        model, 
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for LLaVA on VizWiz dataset.
    - Adds instruction tuned for visually-impaired questioners.
    - Encourages short, clear answers (one word or short phrase).
    - Handles 'unanswerable' gracefully.
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

    images = batch['images']
    questions = batch["questions"]
    expert_context = batch.get("context", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        q_text = (
            "You are helping a visually impaired person understand an image.\n"
            f"Question: {q.strip()}\n"
            "If the question cannot be answered from the image, respond with 'unanswerable'.\n"
            "Otherwise, answer clearly and briefly (a short word or phrase)."
        )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nAdditional visual hints: {ctx.strip()}"

        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"\nPrevious answer attempt: '{draft}'. "
                    "Revise if necessary based on the image."
                )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

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
            decoded = decoded.split("[/INST]")[-1].strip()
            decoded = decoded.split("\n")[0]
            results.append(decoded)
    return results


@torch.no_grad()
def infer_llava_gqa(
        model,
        processor,
        batch,
        max_new_tokens=80,
        return_logits=False,
        return_inputs=False,
    ):
    """
    GQA-optimized inference for LLaVA.
    - Forces concise, single-word or short-phrase answers.
    - Keeps prompts minimal to avoid sentence-style outputs.
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

        q_text = (
            f"Question: {q.strip()}\n"
            "Answer with only one word or a short phrase."
        )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nHint: {ctx.strip()}"

        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += f"\nPrevious answer: {draft.strip()}\nGive only the corrected short answer."

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        # --- generation ---
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
            decoded = decoded.split("[/INST]")[-1].strip()
            decoded = decoded.split("\n")[0]
            # make extra-short postprocess
            decoded = decoded.split(".")[0]
            results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_llava_textvqa(
        model,
        processor,
        batch,
        max_new_tokens=30, 
        return_logits=False,
        return_inputs=False,
    ):
    """
    Improved LLaVA inference for TextVQA.
    - Context-aware prompt (read + interpret text)
    - Cleaned OCR token integration
    - Stable attention masking and decoding
    - Normalized postprocessing
    """

    def _process_image(img):
        """Convert any image input type to RGB PIL Image."""
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

    def _normalize_text(s: str) -> str:
        """Normalize predicted text for fair comparison."""
        s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip().lower()

    images = batch["images"]
    questions = batch["questions"]
    ocr_texts = batch.get("ocr_text", None)
    ocr_tokens = batch.get("ocr_tokens", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        q_text = (
            "You are answering a visual question that may require reading text from the image.\n"
            "Carefully read visible words on signs, boards, or objects, and interpret their meaning.\n"
            "If the text is unclear, infer plausible content using visible context.\n"
            "If no relevant text exists, reply exactly with 'unanswerable'.\n"
            "Provide a concise, factual answer (1–5 words).\n\n"
            f"Question: {q.strip()}"
        )

        if ocr_texts is not None and len(ocr_texts) > i and ocr_texts[i]:
            q_text += f"\nVisible text summary: {ocr_texts[i].strip()}"

        if ocr_tokens is not None and len(ocr_tokens) > i and ocr_tokens[i]:
            token_list = [t.strip() for t in ocr_tokens[i] if len(t.strip()) > 1]
            token_str = ", ".join(sorted(set(token_list)))
            if token_str:
                q_text += f"\nDetected words: {token_str}."

        if draft_answers is not None and len(draft_answers) > i and draft_answers[i]:
            q_text += (
                f"\nPrevious answer attempt: '{draft_answers[i]}'. "
                "Revise it only if inconsistent with the image text."
            )

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q_text}]}
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

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
        decoded = decoded.split("[/INST]")[-1].strip().split("\n")[0]

        decoded_n = _normalize_text(decoded)
        bad = {"", "none", "unknown", "no idea", "nothing", "error", "cant tell"}
        if decoded_n in bad or len(decoded_n) < 2:
            decoded = "unanswerable"
        elif len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_llava_ocrvqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for LLaVA on OCR-VQA (book-cover reading comprehension).
    - Focuses on reading visible text (title, author, genre, etc.)
    - Responds 'unanswerable' if the information cannot be found on the cover.
    - Encourages short factual answers (1–5 words).
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
            f"Question: {q.strip()}\n"
            "Carefully read the visible text on the cover to answer.\n"
            "If the information is not readable or not present, reply exactly with 'unanswerable'.\n"
            "Provide a concise factual answer (1–5 words)."
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
                    f"\nPrevious answer attempt: '{draft}'. "
                    "Correct it if it contradicts the cover text."
                )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue
        elif return_inputs:
            results.append(inputs)
            continue
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            decoded = processor.decode(outputs[0], skip_special_tokens=True)

            decoded = decoded.split("[/INST]")[-1].strip()
            decoded = decoded.split("\n")[0]
            decoded_lower = decoded.lower()

            bad_tokens = ["", "none", "unknown", "no idea", "can't tell", "nothing", "error"]
            if decoded_lower in bad_tokens or len(decoded) < 2:
                decoded = "unanswerable"

            if len(decoded.split()) > 6:
                decoded = " ".join(decoded.split()[:6])

            results.append(decoded.strip())

    return results

    
@torch.no_grad()
def infer_llava_cococaption(
    model,
    processor,
    batch,
    max_new_tokens=25,
    temperature=0.2,
    top_p=0.9,
    return_logits=False,
    return_inputs=False,
):
    """
    Optimized COCO Captioning inference for LLaVA-1.6-Mistral.
    - Follows official prompt style: "[INST] <image>\\nWhat is shown in this image? [/INST]"
    - Uses short deterministic decoding for concise captions.
    - Cleans repetitive or templated outputs.
    """
    def _process_image(img):
        """Convert any image input to RGB PIL."""
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

    images = batch.get("images", [])
    if isinstance(images, (str, torch.Tensor, Image.Image)):
        images = [images]

    results = []
    for img in images:
        proc_img = _process_image(img)
        user_prompt = "Provide a short, neutral English caption describing the main content of the image."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt",
        ).to(model.device)

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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)

        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1].strip()

        for prefix in [
            "caption:", "description:", "the image shows", "this image shows",
            "in this image", "image of", "photo of", "a photo of"
        ]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        decoded = decoded.lstrip(":,.- ").strip()
        results.append(decoded)

    return results


@torch.no_grad()
def infer_llava_nocaps(
    model,
    processor,
    batch,
    max_new_tokens=30,
    temperature=0.3,
    top_p=0.9,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for LLaVA on NoCaps dataset.
    - Generates concise, descriptive English captions.
    - Uses a balanced decoding setup (short, factual, non-repetitive).
    - Cleans prefix patterns and extra tokens for evaluation compatibility.
    """

    def _process_image(img):
        """Convert various input formats to RGB PIL image."""
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

    images = batch.get("images", [])
    if isinstance(images, (str, torch.Tensor, Image.Image)):
        images = [images]

    results = []
    for img in images:
        proc_img = _process_image(img)

        user_prompt = (
            "Generate a short, neutral English caption describing the image. "
            "Focus on the visible objects, scene, and actions only."
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt",
        ).to(model.device)

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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1].strip()

        for prefix in [
            "caption:", "description:", "the image shows", "this image shows",
            "in this image", "image of", "photo of", "a photo of",
            "an image of", "a picture of"
        ]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        decoded = decoded.lstrip(":,.- ").strip()
        decoded = decoded.split("\n")[0]
        results.append(decoded)

    return results


@torch.no_grad()
def infer_llava_flickr(
    model,
    processor,
    batch,
    max_new_tokens=25,
    temperature=0.2,
    top_p=0.9,
    return_logits=False,
    return_inputs=False,
):
    """
    Optimized COCO Captioning inference for LLaVA-1.6-Mistral.
    - Follows official prompt style: "[INST] <image>\\nWhat is shown in this image? [/INST]"
    - Uses short deterministic decoding for concise captions.
    - Cleans repetitive or templated outputs.
    """
    def _process_image(img):
        """Convert any image input to RGB PIL."""
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

    images = batch.get("images", [])
    if isinstance(images, (str, torch.Tensor, Image.Image)):
        images = [images]

    results = []
    for img in images:
        proc_img = _process_image(img)
        user_prompt = "Provide a short, neutral English caption describing the main content of the image."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt",
        ).to(model.device)

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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)

        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1].strip()

        for prefix in [
            "caption:", "description:", "the image shows", "this image shows",
            "in this image", "image of", "photo of", "a photo of"
        ]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        decoded = decoded.lstrip(":,.- ").strip()
        results.append(decoded)

    return results


@torch.no_grad()
def infer_llava_vcr(
    model,
    processor,
    batch,
    max_new_tokens=60,     # shorter generation
    return_logits=False,
    return_inputs=False,
):
    """
    Optimized LLaVA inference for VCR (QA→R, Q→AR)
    - Batched generation for higher GPU utilization
    - Uses torch.compile + caching for speed
    - Parallel postprocessing
    """
    def _process_image(img):
        """Ensure image is RGB PIL image."""
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

    def _clean(text):
        text = text.replace("<end_of_utterance>", "")
        for prefix in ["Answer:", "Response:", "Assistant:", "The answer is"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        return text.strip(" .,:;!?\"'\n\t")

    def _find_most_similar(text, choices):
        text = text.lower().strip()
        scores = [difflib.SequenceMatcher(None, text, c.lower()).ratio() for c in choices]
        return int(torch.tensor(scores).argmax().item())

    # -------- Compile once for speed -------- #
    if not hasattr(model, "_compiled"):
        model = torch.compile(model, mode="reduce-overhead")
        model._compiled = True

    # -------- Prepare batch prompts -------- #
    images = batch["images"]
    questions = batch["questions"]
    answers = batch["answer_choices"]
    rationales = batch["rationales"]
    drafts = batch.get("draft", None)

    batch_images, batch_prompts = [], []
    for i, q in enumerate(questions):
        proc_img = _process_image(images[i])
        ans_text = "\n".join([f"({j+1}) {a}" for j, a in enumerate(answers[i])])
        rat_text = "\n".join([f"({j+1}) {r}" for j, r in enumerate(rationales[i])])

        prompt = (
            f"You are a multimodal reasoning assistant.\n"
            f"Question: {q.strip()}\n"
            f"Answer Choices:\n{ans_text}\n"
            f"Rationale Choices:\n{rat_text}\n"
            "Respond concisely in this format:\n"
            "The best answer is (X) because (Y).\n"
            "End your response with <end_of_utterance>."
        )
        if drafts is not None and len(drafts) > i and drafts[i]:
            prompt += f"\nPrevious answer: {drafts[i]} (Re-evaluate if needed)."

        chat = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )
        batch_images.append(proc_img)
        batch_prompts.append(chat)

    # -------- Tokenize in batch -------- #
    inputs = processor(
        images=batch_images,
        text=batch_prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    if return_inputs:
        input_list = []
        for img, prmpt in zip(batch_images, batch_prompts):
            single_input = processor(
                images=img,
                text=prmpt,
                return_tensors="pt",
            ).to(model.device)
            input_list.append(single_input)
        return input_list

    if return_logits:
        outputs = model(**inputs)
        return outputs.logits

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    decoded_all = processor.batch_decode(outputs, skip_special_tokens=True)
    decoded_all = [d.split("[/INST]")[-1].strip() for d in decoded_all]

    def _extract_one(args):
        decoded, ans_choices, rat_choices = args
        cleaned = _clean(decoded)

        ans_match = re.search(r"answer\s*(?:is|:)?\s*\(?(\d+)\)?", cleaned, re.IGNORECASE)
        rat_match = re.search(r"because\s*\(?(\d+)\)?", cleaned, re.IGNORECASE)

        if ans_match:
            ans_idx = int(ans_match.group(1)) - 1
        else:
            ans_part = cleaned.split("because")[0]
            ans_idx = _find_most_similar(ans_part, ans_choices)

        if rat_match:
            rat_idx = int(rat_match.group(1)) - 1
        else:
            rat_part = cleaned.split("because")[-1]
            rat_idx = _find_most_similar(rat_part, rat_choices)

        ans_idx = max(0, min(len(ans_choices) - 1, ans_idx))
        rat_idx = max(0, min(len(rat_choices) - 1, rat_idx))
        ans_text = ans_choices[ans_idx] if ans_choices else "N/A"
        rat_text = rat_choices[rat_idx] if rat_choices else "N/A"
        return f"Answer {ans_idx + 1}: {ans_text} | Rationale {rat_idx + 1}: {rat_text}"

    with ThreadPoolExecutor() as ex:
        preds = list(ex.map(_extract_one, zip(decoded_all, answers, rationales)))

    return preds


@torch.no_grad()
def infer_llava_vsr(
        model,
        processor,
        batch,
        max_new_tokens=40,
        return_logits=False,
        return_inputs=False,
    ):
    """
    VSR-optimized inference for LLaVA.
    - Reformulates the task as a binary (Yes/No) question about spatial relations.
    - Uses minimal prompting for deterministic accuracy evaluation.
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

    results = []
    for img, q in zip(images, questions):
        proc_img = _process_image(img)
        q_text = f"{q.strip()}\nAnswer only with 'Yes' or 'No'."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        # --- generation / return handling ---
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

        # --- Post-processing: normalize to Yes/No ---
        decoded = decoded.split("[/INST]")[-1].strip()
        decoded = decoded.split("\n")[0].strip().strip(".").lower()

        if decoded.startswith("yes"):
            decoded = "Yes"
        elif decoded.startswith("no"):
            decoded = "No"
        elif "true" in decoded:
            decoded = "Yes"
        elif "false" in decoded:
            decoded = "No"
        else:
            decoded = "No"  # conservative fallback

        results.append(decoded)

    return results


@torch.no_grad()
def infer_llava_okvqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for LLaVA on OK-VQA dataset.
    - Encourages reasoning that combines visual content and commonsense/world knowledge.
    - Produces short factual answers (1–5 words).
    - Supports expert context and draft-based refinement.
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

        q_text = (
            f"Question: {q.strip()}\n"
            "Think about both what is visible in the image and what you know about the world.\n"
            "Use your commonsense or factual knowledge to answer accurately in a short phrase (1–5 words)."
        )
        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nHelpful background knowledge: {ctx.strip()}"
        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"\nPrevious tentative answer: '{draft}'. "
                    "Revise if it seems inconsistent with the image or knowledge."
                )

        # --- build conversation ---
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        # --- generation pipeline ---
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
            decoded = decoded.split("[/INST]")[-1].strip()
            decoded = decoded.split("\n")[0]

            bad_tokens = ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]
            if decoded.lower() in bad_tokens:
                decoded = "unanswerable"

            if len(decoded.split()) > 6:
                decoded = " ".join(decoded.split()[:6])

            # normalize yes/no variants
            low = decoded.lower()
            if low in ["yeah", "yep", "affirmative", "correct"]:
                decoded = "yes"
            elif low in ["nope", "negative", "incorrect"]:
                decoded = "no"

            results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_llava_aokvqa(
    model,
    processor,
    batch,
    max_new_tokens=150,
    mode="MC",  # "DA" (direct-answer) or "MC" (multiple-choice)
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for LLaVA on A-OKVQA dataset.
    - Supports both Multiple-Choice (MC) and Direct-Answer (DA) formats.
    - Encourages grounded reasoning that blends visual and commonsense knowledge.
    - Produces concise factual answers (1–5 words).
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
    choices = batch.get("choices", None)
    expert_context = batch.get("context", None)
    draft = batch.get("draft", None)

    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        if mode == "MC" and choices is not None and len(choices) > i:
            opts = choices[i]
            choice_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(opts)])
            q_text = (
                f"Question: {q.strip()}\n"
                f"Choices:\n{choice_text}\n"
                "Select the most accurate answer (A–E) based on both the image and your knowledge."
            )
        else:
            q_text = (
                f"Question: {q.strip()}\n"
                "Think about both what is visible in the image and what you know about the world.\n"
                "Answer concisely in a short factual phrase (1–5 words)."
            )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nHelpful background knowledge: {ctx.strip()}"

        if draft is not None and len(draft) > i:
            prev = draft[i]
            if prev:
                q_text += (
                    f"\nPrevious tentative answer: '{prev}'. "
                    "Revise only if it conflicts with the image or knowledge."
                )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        # --- generation pipeline ---
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
        decoded = decoded.split("[/INST]")[-1].strip().split("\n")[0]

        # --- normalization ---
        bad_tokens = ["", "unknown", "none", "no idea", "can't tell", "nothing", "error"]
        if decoded.lower() in bad_tokens:
            decoded = "unanswerable"

        # limit to short phrase
        if len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        low = decoded.lower()
        if low in ["yeah", "yep", "affirmative", "correct"]:
            decoded = "yes"
        elif low in ["nope", "negative", "incorrect"]:
            decoded = "no"

        # --- Multiple-choice: map letters to text ---
        if mode == "MC" and choices is not None and len(choices) > i:
            opts = choices[i]
            letter = low.strip().replace(".", "").upper()
            if letter in ["A", "B", "C", "D", "E"]:
                idx = ord(letter) - 65
                if 0 <= idx < len(opts):
                    decoded = opts[idx]

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_llava_sqa(
        model, 
        processor,
        batch,
        max_new_tokens=10,
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
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        return img

    images = batch["images"]
    questions = batch["questions"]
    choices = batch.get("choices", None)
    lectures = batch.get("lectures", None)
    contexts = batch.get("contexts", None)
    drafts = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)
        prompt_parts = []

        if lectures and len(lectures) > i and lectures[i]:
            prompt_parts.append(f"Scientific concept:\n{lectures[i].strip()}")

        if contexts and len(contexts) > i and contexts[i]:
            prompt_parts.append(f"Problem context:\n{contexts[i].strip()}")

        prompt_parts.append(f"Question: {q.strip()}")
        if choices and len(choices) > i and choices[i]:
            chs = choices[i]
            ch_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(chs)])
            prompt_parts.append(f"Choices:\n{ch_text}")

        prompt_parts.append(
            "Answer ONLY with the correct option letter (A, B, C, D, or E). "
            "Do NOT include explanations or reasoning. "
            "If the question involves describing an image, do not describe it."
        )
        q_text = "\n\n".join(prompt_parts)

        if drafts and len(drafts) > i:
            q_text += (
                f"\nThe initial answer was: '{drafts[i]}'. "
                f"Now, given the image evidence, reconsider and provide the corrected answer."
            )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            images=proc_img,
            text=prompt,
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
            temperature=0.0,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        decoded = decoded.split("[/INST]")[-1].strip()

        clean = decoded.strip()
        if len(clean) > 2:
            for c in ["A", "B", "C", "D", "E"]:
                if c in clean:
                    clean = c
                    break

        results.append(clean.strip())

    return results


@torch.no_grad()
def infer_llava_mme(
    model,
    processor,
    batch,
    max_new_tokens=60,
    temperature=0.0,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for LLaVA on MME Benchmark (all yes/no perception tasks).
    - Ensures deterministic 'Yes'/'No' output only.
    - Compatible with all visual reasoning sub-tasks (OCR, color, count, etc.).
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

    def _normalize_output(text: str) -> str:
        """Postprocess to strictly return 'Yes' or 'No'."""
        text = text.lower().strip()
        text = text.replace(".", "").replace(",", "")
        text = text.split("\n")[0]

        # extract yes/no keywords
        if "yes" in text and "no" not in text:
            return "Yes"
        elif "no" in text and "yes" not in text:
            return "No"
        return ""

    images = batch["images"]
    questions = batch["questions"]
    tasks = batch.get("tasks", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)
        task_name = tasks[i] if tasks is not None and len(tasks) > i else "general"

        q_text = (
            f"You are evaluating a {task_name} perception task.\n"
            f"Question: {q.strip()}\n"
            "Answer with a single word — 'Yes' or 'No' — based only on the image."
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
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

        # --- deterministic generation (no sampling) ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1]
        decoded = decoded.strip()
        normalized = _normalize_output(decoded)
        results.append(normalized)

    return results


@torch.no_grad()
def infer_llava_mmbench(
    model,
    processor,
    batch,
    max_new_tokens=60,
    temperature=0.0,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for LLaVA on MMBench Benchmark (multiple-choice tasks).
    - Model chooses between options A/B/C/D.
    - Returns outputs like "A: bike" or "C: piano".
    """

    def _process_image(img):
        """Convert image to RGB PIL."""
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

    def _normalize_choice_output(text, choices_list):
        """
        Normalize output to match one of ['A', 'B', 'C', 'D'] and append actual option text.
        """
        text = text.strip().replace("\n", " ").replace(".", "").lower()

        # find letter in output
        for letter in ["a", "b", "c", "d"]:
            if f"{letter}:" in text or text.startswith(letter) or f"option {letter}" in text:
                for c in choices_list:
                    if c.lower().startswith(letter + ":"):
                        return c 
        return ""

    images = batch["images"]
    questions = batch["questions"]
    choices_list = batch["choices"] 
    tasks = batch.get("tasks", None)

    results = []
    for i, (img, q, chs) in enumerate(zip(images, questions, choices_list)):
        proc_img = _process_image(img)
        task_name = tasks[i] if tasks is not None and len(tasks) > i else "general"

        # --- Build multiple-choice prompt ---
        ch_text = "\n".join(chs)
        q_text = (
            f"You are solving a multiple-choice question in {task_name}.\n"
            f"Question: {q.strip()}\n\n"
            f"Choices:\n{ch_text}\n\n"
            "Answer with only one letter (A, B, C, or D)."
        )

        # --- LLaVA conversation format ---
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
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

        # --- Deterministic generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1]
        decoded = decoded.strip()

        normalized = _normalize_choice_output(decoded, chs)
        results.append(normalized if normalized else decoded)

    return results


@torch.no_grad()
def infer_llava_seedbench(
    model,
    processor,
    batch,
    max_new_tokens=30,
    temperature=0.0,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for LLaVA on SEED-Bench (image-only multiple-choice tasks).
    - Generates deterministic single-letter answers (A/B/C/D).
    - Compatible with various visual reasoning question types.
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

    def _normalize_output(text: str) -> str:
        """
        Normalize model output to one of 'A', 'B', 'C', 'D'.
        """
        text = text.strip().upper().replace(".", "").replace(",", "")
        text = text.split("\n")[0]

        # find explicit answer
        for opt in ["A", "B", "C", "D"]:
            if text.startswith(opt):
                return opt
            if f"{opt}:" in text:
                return opt

        # handle full-word answers
        mapping = {
            "A": ["A", "1", "option A", "answer A"],
            "B": ["B", "2", "option B", "answer B"],
            "C": ["C", "3", "option C", "answer C"],
            "D": ["D", "4", "option D", "answer D"],
        }
        for k, v in mapping.items():
            if any(x in text for x in v):
                return k

        return ""

    images = batch["images"]
    questions = batch["questions"]
    choices_list = batch["choices"]
    qtypes = batch.get("question_types", None)

    results = []

    for i, (img, q, choices) in enumerate(zip(images, questions, choices_list)):
        proc_img = _process_image(img)
        qtype = qtypes[i] if qtypes is not None and len(qtypes) > i else "general"

        # --- Construct SEED-Bench-style prompt ---
        formatted_choices = "\n".join(choices)
        q_text = (
            f"You are solving a {qtype} visual reasoning task.\n"
            f"Question: {q.strip()}\n"
            f"Choices:\n{formatted_choices}\n"
            "Answer only with the letter (A, B, C, or D) corresponding to the correct choice."
        )

        # --- Apply LLaVA conversation template ---
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
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

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]")[-1]
        decoded = decoded.strip()
        normalized = _normalize_output(decoded)

        results.append(normalized)

    return results


@torch.no_grad()
def infer_llava_haloquest(
        model,
        processor,
        batch,
        max_new_tokens=80,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for LLaVA on HaloQuest dataset.
    - Visually grounded and hallucination-resistant prompt.
    - Outputs short factual answers or 'not visible' / 'uncertain' when unsure.
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
        return img

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)
    draft_answers = batch.get("draft", None)

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        prompt = (
            f"Look carefully at the image and answer strictly based on what is visible.\n"
            f"If the object or detail mentioned is not visible, say 'not visible'.\n"
            f"If there is not enough evidence to be certain, say 'uncertain'.\n\n"
            f"Question: {q.strip()}\n"
            f"Answer concisely in one short phrase or sentence."
        )

        # attach context if available
        if expert_context is not None and len(expert_context) > i and expert_context[i]:
            prompt += f"\nExpert context: {expert_context[i]}"

        # attach draft refinement if provided
        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            if draft:
                prompt += f"\nPrevious answer: '{draft}'. Re-evaluate with the image."

        # construct conversation for LLaVA chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=chat_prompt, return_tensors="pt").to(model.device)

        if return_inputs:
            results.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # --- generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        decoded = decoded.split("[/INST]")[-1].strip()

        # --- cleanup ---
        for prefix in ["Answer:", "A:", "Response:", "Output:", "The answer is"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()
        decoded = decoded.strip().capitalize()

        results.append(decoded)

    return results


@torch.no_grad()
def infer_llava_mmhalbench(
    model,
    processor,
    batch,
    max_new_tokens=80,
    temperature=0.0,
    return_logits=False,
    return_inputs=False,
):
    """
    Inference for LLaVA on MMHal-Bench.
    - Supports diverse question types (not limited to Yes/No).
    - Encourages factually grounded, concise answers.
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

    def _postprocess_output(text: str) -> str:
        """Clean excessive tokens or repetition."""
        text = text.strip()
        if "[/INST]" in text:
            text = text.split("[/INST]")[-1]
        text = text.replace("\n", " ").replace("  ", " ").strip()
        return text.split("Answer:")[-1].strip() if "Answer:" in text else text

    images = batch["images"]
    questions = batch["questions"]

    results = []
    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        q_text = (
            f"Question: {q.strip()}\n"
            "Answer concisely and factually based on the image content only."
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        # Build LLaVA-style input
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        if return_inputs:
            results.append(inputs)
            continue

        if return_logits:
            outputs = model(**inputs)
            results.append(outputs.logits)
            continue

        # --- Generate prediction ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        decoded = _postprocess_output(decoded)
        results.append(decoded)

    return results
