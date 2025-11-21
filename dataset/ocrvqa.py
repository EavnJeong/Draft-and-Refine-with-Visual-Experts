import json
import os
from torch.utils.data import Dataset


class OCRVQADataset(Dataset):
    def __init__(self, root_dir, split="testdev", limit=None):
        assert split in ["train", "val", "testdev", "test"], f"Invalid split: {split}"
        self.split = split
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")

        # Path to unflattened metadata (dict keyed by image_id)
        metadata_path = os.path.join(root_dir, "questions", "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Expand each image entry into multiple QA pairs
        questions = []
        for image_id, entry in data.items():
            q_list = entry.get("questions", [])
            a_list = entry.get("answers", [])
            for i, (q, a) in enumerate(zip(q_list, a_list)):
                questions.append({
                    "qid": f"{image_id}_{i}",
                    "imageId": image_id,
                    "question": q,
                    "answer": a,
                    "ocr_tokens": entry.get("ocr_tokens", []),
                    "ocr_info": entry.get("ocr_info", []),
                    "title": entry.get("title", ""),
                    "authorName": entry.get("authorName", ""),
                    "genre": entry.get("genre", ""),
                    "image_path": os.path.join(self.image_dir, f"{image_id}.jpg"),
                })

        if limit is not None:
            questions = questions[:limit]

        self.questions = questions
        print(f"[OCRVQA] Loaded {len(self.questions):,} QA pairs from {os.path.basename(metadata_path)}")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]

        qid = q["qid"]
        image_id = q["imageId"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        return {
            "question_id": qid,
            "image": image_path,
            "question": q["question"],
            "answer": q.get("answer", None),
            "image_id": image_id,
            "ocr_token": q["ocr_tokens"]
        }