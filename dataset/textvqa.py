import os
import json
from torch.utils.data import Dataset


class TextVQADataset(Dataset):
    def __init__(self, root_dir, split="test", limit=None):
        assert split in ["test", "val"], f"Invalid split: {split}"
        self.split = split
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")

        # File names
        filename = f"TextVQA_0.5.1_{split}.json"
        ocr_token_fname = f"TextVQA_Rosetta_OCR_v0.2_{split}.json"
        question_path = os.path.join(root_dir, "questions", filename)
        ocr_path = os.path.join(root_dir, "questions", ocr_token_fname)

        # Load questions
        with open(question_path, "r") as f:
            data = json.load(f)
        self.questions = data["data"]

        # Load OCR annotations
        with open(ocr_path, "r") as f:
            ocr_data = json.load(f)

        # Map image_id → OCR entries
        self.ocr_dict = {}
        for entry in ocr_data["data"]:
            img_id = entry["image_id"]
            self.ocr_dict[img_id] = {
                "tokens": entry.get("ocr_tokens", []),
                "bboxes": entry.get("ocr_info", []),  # optional, if exists
            }
        if limit is not None:
            self.questions = self.questions[:limit]

        print(f"[TextVQA] Loaded {len(self.questions):,} samples with OCR tokens from {filename}")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        qid = q["question_id"]
        image_id = q["image_id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        ocr_entry = self.ocr_dict.get(image_id, {"tokens": [], "bboxes": []})
        ocr_tokens = ocr_entry["tokens"]

        return {
            "question_id": qid,
            "image_id": image_id,
            "image": image_path,
            "question": q["question"],
            "answer": q.get("answers", None),
            "ocr_token": ocr_tokens,
        }