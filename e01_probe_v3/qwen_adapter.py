from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

from models.vlm.loader import load_qwen25_vl


class QwenBinaryScorer:
    """Minimal Qwen2.5-VL binary scorer for causal probes.

    It does not modify the upstream DnR inference path.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        cache_dir: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model, self.processor = load_qwen25_vl(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            dtype=dtype,
        )
        self.model.eval()
        self.model_name = model_name

    @staticmethod
    def _as_rgb(im: Image.Image | str) -> Image.Image:
        if isinstance(im, str):
            return Image.open(im).convert("RGB")
        return im.convert("RGB")

    def _prepare(
        self,
        labeled_images: Sequence[Tuple[str, Image.Image | str]],
        question: str,
        *,
        composite: bool = False,
    ):
        content = []
        images: List[Image.Image] = []

        if composite:
            if len(labeled_images) != 1:
                raise ValueError("Composite mode expects exactly one image")
            content.append({"type": "image"})
            images.append(self._as_rgb(labeled_images[0][1]))
            content.append({
                "type": "text",
                "text": (
                    "The image is a 2x2 composite containing panels labeled A, B, C and D. "
                    + question
                    + " Answer with exactly 0 or 1."
                ),
            })
        else:
            for label, image in labeled_images:
                # Explicit textual anchors are kept constant across all subsets.
                content.append({"type": "text", "text": f"Image slot {label}:"})
                content.append({"type": "image"})
                images.append(self._as_rgb(image))
            content.append({
                "type": "text",
                "text": question + " Answer with exactly 0 or 1.",
            })

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        kwargs = dict(text=prompt, return_tensors="pt")
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs)
        inputs = {
            k: (v.to(self.model.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }
        return prompt, inputs

    def _raw_candidate_logprob(self, inputs, candidate: str) -> float:
        tokenizer = self.processor.tokenizer
        ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if not ids:
            raise RuntimeError(f"Candidate tokenization is empty: {candidate!r}")

        base_ids = inputs["input_ids"]
        base_len = base_ids.shape[1]
        cand = torch.tensor([ids], dtype=base_ids.dtype, device=base_ids.device)

        run_inputs = dict(inputs)
        run_inputs["input_ids"] = torch.cat([base_ids, cand], dim=1)
        if "attention_mask" in inputs:
            extra = torch.ones(
                (inputs["attention_mask"].shape[0], len(ids)),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            )
            run_inputs["attention_mask"] = torch.cat([inputs["attention_mask"], extra], dim=1)

        out = self.model(**run_inputs)
        logp = torch.log_softmax(out.logits.float(), dim=-1)
        score = 0.0
        for j, token_id in enumerate(ids):
            score += float(logp[0, base_len - 1 + j, token_id].item())
        return score

    @torch.no_grad()
    def score_binary(
        self,
        labeled_images: Sequence[Tuple[str, Image.Image | str]],
        question: str,
        *,
        composite: bool = False,
    ) -> Dict[str, float | str]:
        prompt, inputs = self._prepare(labeled_images, question, composite=composite)

        raw0 = self._raw_candidate_logprob(inputs, "0")
        raw1 = self._raw_candidate_logprob(inputs, "1")
        m = max(raw0, raw1)
        z = math.exp(raw0 - m) + math.exp(raw1 - m)
        logz = m + math.log(z)
        logp0 = raw0 - logz
        logp1 = raw1 - logz
        return {
            "raw_logp_0": raw0,
            "raw_logp_1": raw1,
            "logp_0": logp0,
            "logp_1": logp1,
            "p_0": math.exp(logp0),
            "p_1": math.exp(logp1),
            "prediction": "1" if logp1 > logp0 else "0",
            "prompt": prompt,
        }

    @torch.no_grad()
    def score_binary_text(self, question: str) -> Dict[str, float | str]:
        content = [{"type": "text", "text": question + " Answer with exactly 0 or 1."}]
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }
        raw0 = self._raw_candidate_logprob(inputs, "0")
        raw1 = self._raw_candidate_logprob(inputs, "1")
        m = max(raw0, raw1)
        z = math.exp(raw0 - m) + math.exp(raw1 - m)
        logz = m + math.log(z)
        return {
            "raw_logp_0": raw0,
            "raw_logp_1": raw1,
            "logp_0": raw0 - logz,
            "logp_1": raw1 - logz,
            "p_0": math.exp(raw0 - logz),
            "p_1": math.exp(raw1 - logz),
            "prediction": "1" if raw1 > raw0 else "0",
            "prompt": prompt,
        }
