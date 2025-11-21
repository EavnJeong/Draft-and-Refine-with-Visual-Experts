import json
import os
from torch.utils.data import Dataset


class GQADataset(Dataset):
    def __init__(self, root_dir, split="testdev", balanced=True, limit=None):
        assert split in ["train", "val", "testdev", "test"], f"Invalid split: {split}"
        self.split = split
        self.root_dir = root_dir
        self.balanced = balanced
        self.image_dir = os.path.join(root_dir, "images")

        filename = f"{split}_{'balanced' if balanced else 'all'}_questions.json"
        question_path = os.path.join(root_dir, "questions", filename)

        with open(question_path, "r") as f:
            data = json.load(f)

        # Keep both question id and content
        self.questions = [{"qid": k, **v} for k, v in data.items()]

        if limit is not None:
            self.questions = self.questions[:limit]

        print(f"[GQA] Loaded {len(self.questions):,} samples from {filename}")

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
        }