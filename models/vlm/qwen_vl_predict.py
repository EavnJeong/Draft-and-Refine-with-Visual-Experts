import torch
import difflib
from PIL import Image
import re
from difflib import SequenceMatcher


@torch.no_grad()
def infer_qwen(
        model,
        processor,
        batch,
        max_new_tokens=200,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

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
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)
            decoded = decoded[0].split('\nassistant\n')[-1].strip()
            results.append(decoded.strip())
    return results


@torch.no_grad()
def infer_qwen_vizwiz(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on VizWiz dataset.
    - Tailored prompt for visually-impaired users' questions.
    - Encourages short, clear answers or 'unanswerable' when uncertain.
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
            "You are helping a visually impaired person understand an image.\n"
            f"Question: {q.strip()}\n"
            "If the image is unclear, unreadable, or does not contain enough information, "
            "respond with 'unanswerable'. Otherwise, give a short and clear answer "
            "(one or a few words)."
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
                    "Revise it if the image provides better evidence."
                )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

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

            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            decoded = decoded.split("\nassistant\n")[-1].strip()
            decoded = decoded.split("Assistant:")[-1].strip()
            decoded = decoded.replace("<|im_end|>", "").strip()
            decoded = decoded.split("\n")[0]

            if not decoded:
                decoded = "unanswerable"

            results.append(decoded)
    return results


@torch.no_grad()
def infer_qwen_gqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
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

        q_text = (
            "You are answering a visual reasoning question about the image.\n"
            f"Question: {q.strip()}\n"
            "Provide a short and precise answer (a single word or short phrase). "
            "Do not explain your reasoning."
        )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nScene hints: {ctx.strip()}"

        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"\nInitial hypothesis: '{draft}'. "
                    "Verify this answer based on the image and correct it if necessary."
                )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

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

            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            decoded = decoded.split("\nassistant\n")[-1].strip()
            decoded = decoded.split("Assistant:")[-1].strip()
            decoded = decoded.replace("<|im_end|>", "").strip()
            decoded = decoded.split("\n")[0]

            if not decoded:
                decoded = "unknown"

            results.append(decoded)
    return results


@torch.no_grad()
def infer_qwen_textvqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on TextVQA dataset.
    - Designed for text-reading tasks (signs, documents, labels, etc.).
    - Supports optional OCR text and OCR tokens for better grounding.
    - Encourages concise factual answers (1–5 words).
    - Replies 'unanswerable' when text is unreadable or missing.
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
            "You are answering a question that requires reading visible text in an image.\n"
            f"Question: {q.strip()}\n"
            "Carefully examine all signs, labels, and printed text. "
            "If the text is unreadable or the answer cannot be inferred, reply exactly with 'unanswerable'.\n"
            "Otherwise, answer concisely in one or a few words."
        )

        context_str = ""
        if ocr_texts is not None and len(ocr_texts) > i and ocr_texts[i]:
            context_str += f"Recognized text from OCR: {ocr_texts[i].strip()}"
        if ocr_tokens is not None and len(ocr_tokens) > i and ocr_tokens[i]:
            tokens_str = " | ".join(ocr_tokens[i])
            context_str += ("\n" if context_str else "") + f"Detected tokens: {tokens_str}"

        if context_str:
            q_text += f"\n{context_str}"

        if draft_answers is not None and len(draft_answers) > i:
            draft = draft_answers[i]
            if draft:
                q_text += (
                    f"\nPrevious answer attempt: '{draft}'. "
                    "Revise it if inconsistent with the text in the image."
                )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

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
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        decoded = decoded.split("\nassistant\n")[-1].strip()
        decoded = decoded.split("Assistant:")[-1].strip()
        decoded = decoded.replace("<|im_end|>", "").strip()
        decoded = decoded.split("\n")[0]

        decoded_lower = decoded.lower()
        bad_tokens = ["", "none", "unknown", "can't tell", "illegible", "no idea", "error"]
        if decoded_lower in bad_tokens or len(decoded) < 2:
            decoded = "unanswerable"

        if len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_qwen_ocrvqa(
        model,
        processor,
        batch,
        max_new_tokens=150,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on OCR-VQA (book-cover reading).
    - Focused on extracting information from visible text.
    - Produces short factual answers (title, author, genre, etc.).
    - Returns 'unanswerable' when the cover text does not contain the answer.
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
            "Read the visible text on the cover carefully.\n"
            "If the information is missing, illegible, or cannot be found, "
            "reply exactly with 'unanswerable'. Otherwise, answer concisely "
            "(1–5 words)."
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
                    "Revise it if the cover text provides a better answer."
                )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

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
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        decoded = decoded.split("\nassistant\n")[-1].strip()
        decoded = decoded.split("Assistant:")[-1].strip()
        decoded = decoded.replace("<|im_end|>", "").strip()
        decoded = decoded.split("\n")[0]

        decoded_lower = decoded.lower()
        bad_tokens = ["", "none", "unknown", "no idea", "can't tell", "nothing", "error"]
        if decoded_lower in bad_tokens or len(decoded) < 2:
            decoded = "unanswerable"
        if decoded.lower().startswith("yes"):
            decoded = "Yes"
        elif decoded.lower().startswith("no"):
            decoded = "No"

        if len(decoded.split()) > 6:
            decoded = " ".join(decoded.split()[:6])

        results.append(decoded.strip())

    return results


@torch.no_grad()
def infer_qwen_cococaption(
        model,
        processor,
        batch,
        max_new_tokens=40,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on COCO Captioning.
    - Generates concise captions describing the image.
    - Removes conversational prefixes from output.
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
    results = []

    for i, img in enumerate(images):
        proc_img = _process_image(img)

        q_text = (
            "Describe the given image with a single, concise English sentence.\n"
            "Avoid phrases like 'This image shows' or 'The picture of'."
        )

        if "context" in batch and len(batch["context"]) > i and batch["context"][i]:
            q_text += f"\nAdditional visual context: {batch['context'][i].strip()}"

        if "draft" in batch and len(batch["draft"]) > i and batch["draft"][i]:
            q_text += (
                f"\nInitial caption suggestion: '{batch['draft'][i]}'. "
                "Refine it based on visual evidence."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

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

            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            decoded = decoded.split("\nassistant\n")[-1].strip()
            decoded = decoded.replace("Assistant:", "").replace("<|im_end|>", "").strip()

            if decoded and decoded[0].islower():
                decoded = decoded[0].upper() + decoded[1:]
            if not decoded.endswith("."):
                decoded += "."

            results.append(decoded)
    return results


@torch.no_grad()
def infer_qwen_nocaps(
        model,
        processor,
        batch,
        max_new_tokens=40,
        temperature=0.25,
        top_p=0.85,
        return_logits=False,
        return_inputs=False,
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

    # ------------------------------
    images = batch["images"]
    expert_context = batch.get("context", None)
    draft = batch.get("draft", None)

    results = []

    for i, img in enumerate(images):
        proc_img = _process_image(img)

        q_text = (
            "Describe this image in a short, factual English sentence "
            "(no background, no emotions, no storytelling). "
            "Focus only on main objects and their actions or relationships."
        )

        if expert_context is not None and len(expert_context) > i and expert_context[i]:
            q_text += f"\nVisual hints: {expert_context[i].strip()}"

        if draft is not None and len(draft) > i and draft[i]:
            q_text += (
                f"\nPrevious caption: '{draft[i].strip()}'. "
                "Make it more factual and concise."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        decoded = decoded.split("\nassistant\n")[-1]
        decoded = decoded.replace("Assistant:", "").replace("<|im_end|>", "")
        decoded = decoded.strip()

        decoded = re.sub(
            r"\b(scene|setting|picture|image|photo|background|outdoors|indoors|environment|area|surroundings|amidst|in front of|near|behind)\b",
            "",
            decoded,
            flags=re.IGNORECASE,
        )
        decoded = re.sub(r"\bappears to|seems to|looks like\b", "", decoded, flags=re.IGNORECASE)
        decoded = re.sub(r"\s+", " ", decoded).strip()

        decoded = " ".join(decoded.split()[:15])

        if decoded and decoded[0].islower():
            decoded = decoded[0].upper() + decoded[1:]
        if not decoded.endswith("."):
            decoded += "."

        results.append(decoded)

    return results
    

@torch.no_grad()
def infer_qwen_flickr(
        model,
        processor,
        batch,
        max_new_tokens=40,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on COCO Captioning.
    - Generates concise captions describing the image.
    - Removes conversational prefixes from output.
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
    results = []

    for i, img in enumerate(images):
        proc_img = _process_image(img)

        q_text = (
            "Describe the given image with a single, concise English sentence.\n"
            "Avoid phrases like 'This image shows' or 'The picture of'."
        )

        if "context" in batch and len(batch["context"]) > i and batch["context"][i]:
            q_text += f"\nAdditional visual context: {batch['context'][i].strip()}"

        if "draft" in batch and len(batch["draft"]) > i and batch["draft"][i]:
            q_text += (
                f"\nInitial caption suggestion: '{batch['draft'][i]}'. "
                "Refine it based on visual evidence."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

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

            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            decoded = decoded.split("\nassistant\n")[-1].strip()
            decoded = decoded.replace("Assistant:", "").replace("<|im_end|>", "").strip()

            if decoded and decoded[0].islower():
                decoded = decoded[0].upper() + decoded[1:]
            if not decoded.endswith("."):
                decoded += "."

            results.append(decoded)
    return results


@torch.no_grad()
def infer_qwen_vcr(
    model,
    processor,
    batch,
    max_new_tokens=150,
    return_logits=False,
    return_inputs=False,
):
    """
    Qwen2-VL inference for Visual Commonsense Reasoning (VCR)
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

    def _clean(text):
        text = text.replace("<|im_end|>", "").strip()
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1]
        return text.strip(" .,:;!?\"'\n\t")

    def _find_most_similar(text, choices):
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
        proc_img = _process_image(images[i])
        ans_text = "\n".join([f"({j+1}) {a}" for j, a in enumerate(answers[i])])
        rat_text = "\n".join([f"({j+1}) {r}" for j, r in enumerate(rationales[i])])

        # --- Unified ViCor-style prompt ---
        user_prompt = (
            "You are an expert at visual commonsense reasoning.\n"
            "Analyze the image carefully and answer the question.\n"
            f"Question: {q.strip()}\n"
            f"Answer Choices:\n{ans_text}\n"
            f"Rationale Choices:\n{rat_text}\n"
            "Select the best answer and rationale.\n"
            "Respond concisely in this format:\n"
            "The best answer is (X) because (Y).\n"
            "End with <end_of_utterance>.\n"
        )

        if drafts is not None and len(drafts) > i and drafts[i]:
            user_prompt += (
                f"The previous answer was '{drafts[i]}'. "
                "Re-evaluate and correct if necessary.\n"
            )

        # --- Build multimodal chat template ---
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

        # --- Early exits ---
        if return_inputs:
            preds.append(inputs)
            continue
        if return_logits:
            outputs = model(**inputs)
            preds.append(outputs.logits)
            continue

        # --- Generation ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
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
def infer_qwen_vsr(
        model,
        processor,
        batch,
        max_new_tokens=60,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL (or similar Qwen-VL) models on the Visual Spatial Reasoning (VSR) dataset.
    - Reformulates each sample as a binary Yes/No reasoning question.
    - Ensures deterministic, concise outputs for accuracy-based evaluation.
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

        # --- concise VSR prompt ---
        q_text = (
            f"{q.strip()}\n"
            "Answer only with 'Yes' or 'No'. Do not explain."
        )

        # --- format for Qwen chat template ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = processor(
            images=proc_img,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        # --- inference mode ---
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

        # --- postprocessing ---
        decoded = decoded.split("\nassistant\n")[-1].strip()
        decoded = decoded.split("Assistant:")[-1].strip()
        decoded = decoded.replace("<|im_end|>", "").strip()
        decoded = decoded.split("\n")[0].lower()

        # normalize binary responses
        if decoded.startswith("yes"):
            decoded = "Yes"
        elif decoded.startswith("no"):
            decoded = "No"
        elif "true" in decoded:
            decoded = "Yes"
        elif "false" in decoded:
            decoded = "No"
        else:
            decoded = "No"  # fallback default

        results.append(decoded)

    return results


@torch.no_grad()
def infer_qwen_okvqa(
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

    def normalize_answer(ans: str) -> str:
        ans = ans.lower().strip()
        ans = re.sub(r"[^a-z0-9\s]", "", ans)
        ans = re.sub(r"\b(a|an|the)\b", "", ans)
        ans = re.sub(r"\s+", " ", ans)
        return ans.strip()

    def fuzzy_match(pred, gts, threshold=0.85):
        pred_norm = normalize_answer(pred)
        for gt in gts:
            gt_norm = normalize_answer(gt)
            if SequenceMatcher(None, pred_norm, gt_norm).ratio() > threshold:
                return True
        return False

    SYN_MAP = {
        "motorcycle racing": "race",
        "horse racing": "race",
        "surfboard": "surf board",
        "bus": "public transport",
        "computer": "pc",
        "cell phone": "phone",
        "soccer": "football",
        "motorbike": "motorcycle",
        "surfing": "surf",
        "polo sport": "polo",
    }

    def postprocess_answer(pred: str) -> str:
        pred = normalize_answer(pred)
        if pred in ["", "unknown", "none", "no idea", "cant tell", "nothing"]:
            return "unanswerable"
        for k, v in SYN_MAP.items():
            if k in pred:
                return v
        return pred

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)
    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        q_text = (
            f"Question: {q.strip()}\n"
            "Answer briefly with one or two words only."
        )

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"\nContext: {ctx.strip()}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
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
            temperature=0.0,  
            top_p=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded = decoded.split("\nassistant\n")[-1].strip()
        decoded = decoded.replace("<|im_end|>", "").strip()
        decoded = decoded.split("Assistant:")[-1].strip()
        decoded = decoded.split("\n")[0]

        final_answer = postprocess_answer(decoded)
        results.append(final_answer)

    return results


@torch.no_grad()
def infer_qwenvl_aokvqa(
    model,
    processor,
    batch,
    max_new_tokens=80,
    mode="MC",  # "MC" (Multiple-Choice) or "DA" (Direct-Answer)
    return_logits=False,
    return_inputs=False,
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

    def normalize_answer(ans: str) -> str:
        ans = ans.lower().strip()
        ans = re.sub(r"[^a-z0-9\s]", "", ans)
        ans = re.sub(r"\b(a|an|the)\b", "", ans)
        ans = re.sub(r"\s+", " ", ans)
        return ans.strip()

    SYN_MAP = {
        "public transport": "bus",
        "motorcycle racing": "race",
        "horse racing": "race",
        "surfboard": "surf board",
        "cell phone": "phone",
        "soccer": "football",
        "motorbike": "motorcycle",
        "surfing": "surf",
        "polo sport": "polo",
        "visibility for safety": "reflector",
    }

    def postprocess_answer(pred: str) -> str:
        pred = normalize_answer(pred)
        if pred in ["", "unknown", "none", "no idea", "cant tell", "nothing"]:
            return "unanswerable"
        for k, v in SYN_MAP.items():
            if k in pred:
                return v
        return pred

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
            "Use both the image and your world knowledge to answer accurately.\n"
            f"Question: {q.strip()}"
        )

        choice_labels = []
        if mode == "MC" and choices is not None and len(choices) > i:
            opts = choices[i]
            if opts not in [None, "", []]:
                choice_labels = [chr(65 + j) for j in range(len(opts))]
                opt_text = "\n".join([f"({label}) {text}" for label, text in zip(choice_labels, opts)])
                prompt += (
                    f"\nOptions:\n{opt_text}\n"
                    "Choose the most accurate option and respond ONLY with its content (not the letter). "
                    "Use at most 3 words."
                )
        else:
            prompt += "\nAnswer briefly in 1–3 words, no sentences."

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx and len(ctx.strip()) > 0:
                prompt += f"\nHelpful background: {ctx.strip()}"
        if draft_answers is not None and len(draft_answers) > i:
            d = draft_answers[i]
            if d and len(d.strip()) > 0:
                prompt += f"\nPrevious tentative answer: '{d.strip()}'. Refine if necessary."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=chat_prompt, return_tensors="pt").to(model.device)

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
            temperature=0.0,
            top_p=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded = decoded.split("\nassistant\n")[-1].strip()
        decoded = decoded.replace("<|im_end|>", "").strip()
        decoded = decoded.split("Assistant:")[-1].strip()
        decoded = decoded.split("\n")[0].strip()

        if len(decoded) == 1 and decoded.upper() in choice_labels:
            idx = ord(decoded.upper()) - 65
            if 0 <= idx < len(choices[i]):
                decoded = choices[i][idx]

        if re.match(r"^[A-Da-d]\s", decoded):
            decoded = decoded[2:].strip()

        decoded = re.split(r"[.,!?]", decoded)[0].strip()
        decoded = " ".join(decoded.split()[:3])  # limit to 3 words

        final_answer = postprocess_answer(decoded)
        results.append(final_answer)

    return results


@torch.no_grad()
def infer_qwen_sqa(
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

    images = batch["images"]
    questions = batch["questions"]
    choices = batch.get("choices", None)
    contexts = batch.get("contexts", None)
    lectures = batch.get("lectures", None)
    expert_context = batch.get("context", None)

    results = []
    for i, img in enumerate(images):
        proc_img = _process_image(img)
        q = questions[i]

        q_text = "You are solving a science multiple-choice question.\n"

        if lectures is not None and len(lectures) > i and lectures[i]:
            q_text += f"Lecture: {lectures[i].strip()}\n"
        if contexts is not None and len(contexts) > i and contexts[i]:
            q_text += f"Context: {contexts[i].strip()}\n"

        q_text += f"Question: {q.strip()}\n"
        if choices is not None and len(choices) > i:
            opts = choices[i]
            formatted_opts = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(opts)])
            q_text += f"Choices:\n{formatted_opts}\n"

        if expert_context is not None and len(expert_context) > i:
            ctx = expert_context[i]
            if ctx:
                q_text += f"Additional clue: {ctx.strip()}\n"

        if "draft" in batch and len(batch["draft"]) > i:
            draft = batch["draft"][i]
            if draft:
                q_text += (
                    f"The previous tentative answer was '{draft}'. "
                    "Reconsider this answer using the image and context, "
                    "and output the final answer as a single letter (A/B/C/D/E)."
                )
            else:
                q_text += "Provide only the letter (A/B/C/D/E) of the correct answer."

        else:
            q_text += "Provide only the letter (A/B/C/D/E) of the correct answer."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

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

            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            decoded = decoded.split("\nassistant\n")[-1].strip()
            decoded = decoded.split("Assistant:")[-1].strip()
            decoded = decoded.replace("<|im_end|>", "").strip()
            decoded = decoded.split("\n")[0]

            match = re.search(r"\b([A-E])\b", decoded, re.IGNORECASE)
            if match:
                decoded = match.group(1).upper()
            else:
                decoded = decoded.strip().split()[0] if decoded else "A"

            results.append(decoded)
    return results


@torch.no_grad()
def infer_qwen_mme(
        model,
        processor,
        batch,
        max_new_tokens=60,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on MME Benchmark (Yes/No tasks).
    Ensures deterministic, clean 'Yes' or 'No' outputs based on image evidence.
    """

    # --- Image normalization ---
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

    def postprocess_yesno(decoded_texts):
        """Clean and normalize model outputs to 'Yes'/'No'."""
        cleaned = []
        for t in decoded_texts:
            t = t.replace("<|im_end|>", "")
            t = t.replace("Assistant:", "")
            t = t.split("\nassistant\n")[-1]
            t = t.strip().lower()

            if re.search(r"\byes\b", t) and not re.search(r"\bno\b", t):
                cleaned.append("Yes")
            elif re.search(r"\bno\b", t) and not re.search(r"\byes\b", t):
                cleaned.append("No")
            else:
                cleaned.append("")  # fallback when unclear
        return cleaned

    # --- Prepare inputs ---
    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    tasks = batch.get("tasks", None)

    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        task_name = tasks[i] if tasks and len(tasks) > i else "general"

        prompt = (
            f"Task: {task_name}\n"
            f"Question: {q.strip()}\n"
            "Answer strictly with only one word: 'Yes' or 'No'."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = processor(
            images=img,
            text=chat_prompt,
            return_tensors="pt",
        ).to(model.device)

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
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        cleaned = postprocess_yesno(decoded)
        results.extend(cleaned)

    return results


@torch.no_grad()
def infer_qwen_mmbench(
        model,
        processor,
        batch,
        max_new_tokens=80,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on MMBench (multiple-choice tasks).
    Generates deterministic choice labels (A/B/C/D) based on visual evidence.
    """

    # --- Image normalization ---
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

    # --- Output cleaning ---
    def postprocess_choice(decoded_texts):
        """
        Normalize model outputs to one of ['A', 'B', 'C', 'D'].
        Falls back to '' if ambiguous.
        """
        results = []
        for t in decoded_texts:
            t = t.replace("<|im_end|>", "")
            t = t.replace("Assistant:", "")
            t = t.split("\nassistant\n")[-1].strip()

            # Extract first clear choice letter
            match = re.search(r"\b([A-D])\b", t, re.IGNORECASE)
            if match:
                results.append(match.group(1).upper())
            else:
                results.append("")
        return results

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    choices_list = batch.get("choices", None)
    tasks = batch.get("tasks", None)

    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        task_name = tasks[i] if tasks and len(tasks) > i else "MMBench"

        # --- Build multiple-choice prompt ---
        if choices_list and len(choices_list) > i:
            choice_lines = "\n".join(choices_list[i])
            prompt = (
                f"Task: {task_name}\n"
                f"Question: {q.strip()}\n"
                f"{choice_lines}\n"
                "Answer with only one letter (A, B, C, or D)."
            )
        else:
            prompt = (
                f"Task: {task_name}\n"
                f"Question: {q.strip()}\n"
                "Answer with only one letter (A, B, C, or D)."
            )

        # --- Construct multimodal chat message ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = processor(
            images=img,
            text=chat_prompt,
            return_tensors="pt",
        ).to(model.device)

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
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        cleaned = postprocess_choice(decoded)
        results.extend(cleaned)

    return results


@torch.no_grad()
def infer_qwen_seedbench(
        model,
        processor,
        batch,
        max_new_tokens=40,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on SEED-Bench (multiple-choice tasks).
    - Handles 4-choice (A/B/C/D) format.
    - Cleans conversational artifacts.
    - Returns the predicted choice letter only (A/B/C/D or empty string if uncertain).
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

    def postprocess_choice(decoded_texts):
        """Normalize output to one of ['A', 'B', 'C', 'D']."""
        cleaned = []
        for t in decoded_texts:
            t = t.replace("<|im_end|>", "")
            t = t.replace("Assistant:", "")
            t = t.split("\nassistant\n")[-1]
            t = t.strip().upper()

            # Extract letter
            match = re.search(r"\b([ABCD])\b", t)
            if match:
                cleaned.append(match.group(1))
            else:
                cleaned.append("")  # fallback
        return cleaned

    images = [_process_image(img) for img in batch["images"]]
    questions = batch["questions"]
    choices = batch["choices"]
    qtypes = batch.get("question_types", None)
    qids = batch.get("question_ids", None)

    results = []

    for i, (img, q, chs) in enumerate(zip(images, questions, choices)):
        qtype_name = qtypes[i] if qtypes and len(qtypes) > i else "general"

        # Construct prompt
        formatted_choices = "\n".join(chs)
        prompt = (
            f"Task: {qtype_name}\n"
            f"Question: {q.strip()}\n"
            f"{formatted_choices}\n"
            "Answer with only the letter (A, B, C, or D)."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = processor(
            images=img,
            text=chat_prompt,
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
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        cleaned = postprocess_choice(decoded)
        results.extend(cleaned)

    return results


@torch.no_grad()
def infer_qwen_haloquest(
        model,
        processor,
        batch,
        max_new_tokens=80,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL on HaloQuest dataset.
    - Simple and grounded prompt (no few-shot examples)
    - Allows 'not visible' / 'uncertain' / 'unanswerable' type answers
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

    images = batch["images"]
    questions = batch["questions"]
    expert_context = batch.get("context", None)

    results = []

    for i, (img, q) in enumerate(zip(images, questions)):
        proc_img = _process_image(img)

        # --- concise but safe prompt ---
        q_text = (
            f"Look carefully at the image and answer only based on what is clearly visible.\n"
            f"If the object or detail is not visible, reply 'not visible'.\n"
            f"If unsure or unclear, reply 'uncertain'.\n\n"
            f"Question: {q.strip()}\nAnswer concisely:"
        )

        if expert_context is not None and len(expert_context) > i and expert_context[i]:
            q_text += f"\n(Extra context: {expert_context[i]})"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
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

        # --- generate output ---
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded = decoded.split("\nassistant\n")[-1].strip()

        # clean-up
        for prefix in ["Answer:", "A:", "Response:", "Output:", "The answer is"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()
        decoded = decoded.strip().capitalize()

        results.append(decoded)

    return results


@torch.no_grad()
def infer_qwen_mmhalbench(
        model,
        processor,
        batch,
        max_new_tokens=60,
        return_logits=False,
        return_inputs=False,
    ):
    """
    Inference for Qwen2.5-VL models on MMHal-Bench dataset.
    - Model is NOT informed about the question type.
    - Encourages concise, factual, grounded answers.
    - Uses fallback answers ('not visible', 'uncertain') when visual evidence is insufficient.
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

        # --- Unified neutral prompt (no qtype info) ---
        q_text = (
            "Answer the question based only on what is visible in the image.\n"
            f"Question: {q.strip()}\n"
            "If the content is unclear or missing, respond with 'uncertain' or 'not visible'."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_text},
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=proc_img, text=prompt, return_tensors="pt").to(model.device)

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

        # --- Postprocess ---
        decoded = decoded.split("\nassistant\n")[-1].strip()
        decoded = decoded.replace("<|im_end|>", "").strip()
        decoded = decoded.split("Assistant:")[-1].strip()
        decoded = decoded.split("\n")[0].strip()

        for prefix in ["Answer:", "Response:", "A:", "Output:", "The answer is"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        results.append(decoded.capitalize())

    return results