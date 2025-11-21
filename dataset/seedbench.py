import os
import json
from torch.utils.data import Dataset
from collections import defaultdict


class SEEDBenchDataset(Dataset):
    """
    SEED-Bench (image-only) Dataset Loader
    - Returns dict: {question, choices, answer, image_path, question_type, question_id}
    """

    def __init__(self, root_dir, limit_per_type=19, verify_images=True):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.json_path = os.path.join(root_dir, "SEED-Bench.json")

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Missing file: {self.json_path}")

        with open(self.json_path, "r") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        qtype_map = data.get("question_type", {})

        # Build reverse map: id -> name
        id2type = {v: k for k, v in qtype_map.items()}

        self.samples = []
        type_counter = defaultdict(int)

        for q in questions:
            if q.get("data_type") != "image":
                continue  # only keep image-type

            qtype_name = id2type.get(q.get("question_type_id"), "Unknown")
            if limit_per_type and type_counter[qtype_name] >= limit_per_type:
                continue

            data_id = q.get("data_id", "")
            image_path = os.path.join(self.image_dir, data_id)

            if verify_images and not os.path.exists(image_path):
                print(f"[Warning] Missing image: {image_path}")
                continue

            def safe_strip(x):
                return x.strip() if isinstance(x, str) else ""

            choices = [
                f"A: {safe_strip(q.get('choice_a'))}",
                f"B: {safe_strip(q.get('choice_b'))}",
                f"C: {safe_strip(q.get('choice_c'))}",
                f"D: {safe_strip(q.get('choice_d'))}",
            ]

            self.samples.append({
                "question_id": str(q.get("question_id", "")),
                "question": safe_strip(q.get("question")),
                "choices": choices,
                "answer": safe_strip(q.get("answer")),
                "image_path": image_path,
                "question_type": qtype_name,
            })

            type_counter[qtype_name] += 1

        print(f"[Loaded SEED-Bench] {len(self.samples)} image samples from {self.json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]