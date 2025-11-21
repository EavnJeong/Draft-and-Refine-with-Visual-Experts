import json
import os
from torch.utils.data import Dataset


class AOKVQADataset(Dataset):
    def __init__(self, root_dir, split="val", limit=None):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.root_dir = root_dir
        self.split = split

        json_path = os.path.join(root_dir, f"aokvqa_v1p0_{split}.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        self.samples = []
        for item in data:
            image_id = item["image_id"]
            question_id = item["question_id"]
            question = item["question"]
            answers = item.get("direct_answers", [])
            mc_answer = item.get("multiple_choice_answer", None)
            choices = item.get("choices", None)

            image_path = os.path.join(
                root_dir, "images", f"{split}2017", f"{image_id:012d}.jpg"
            )

            self.samples.append({
                "question_id": question_id,
                "image_id": image_id,
                "image": image_path,
                "question": question,
                "answers": answers,
                "multiple_choice_answer": mc_answer,
                "choices": choices,
            })

        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]