import json
import os
from torch.utils.data import Dataset


class ScienceQADataset(Dataset):
    def __init__(self, root_dir, split="train", options=None, limit=None):
        self.root_dir = root_dir
        self.split = split
        self.options = options or ["A", "B", "C", "D", "E"]

        problems_path = os.path.join(root_dir, "problems.json")
        splits_path = os.path.join(root_dir, "pid_splits.json")

        with open(problems_path, "r") as f:
            self.problems = json.load(f)
        with open(splits_path, "r") as f:
            split_ids = json.load(f)[split]

        split_ids = [pid for pid in split_ids if self.problems[pid].get("image") is not None]

        if limit is not None:
            split_ids = split_ids[:limit]

        self.samples = {pid: self.problems[pid] for pid in split_ids}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid = list(self.samples.keys())[idx]
        prob = self.samples[pid]

        image = None
        if prob.get('image') is not None:
            image = os.path.join(self.root_dir, "images", f"{self.split}/{pid}", prob['image'])

        question = prob["question"]
        choices = prob["choices"]
        answer_idx = prob["answer"]
        answer = self.options[answer_idx]

        context = prob.get("hint", "")
        lecture = prob.get("lecture", "")
        solution = prob.get("solution", "")

        info = {
            "image": image,
            "question_id": pid,
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "answer": answer,
            "context": context,
            "lecture": lecture,
            "solution": solution,
        }
        return info