import os
import json
from torch.utils.data import Dataset


class VizwizDataset(Dataset):
    def __init__(self, root_dir, ann_file, limit=None):
        super().__init__()
        self.root_dir = root_dir
        self.samples = self.load_annotations(ann_file)
        if limit is not None:
            self.samples = self.samples[:limit]

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)

        samples = []
        for d in data:
            image_path = os.path.join(self.root_dir, d["image"])
            question = d["question"]
            answers = [a["answer"] for a in d.get("answers", [])]
            answer_type = d.get("answer_type", None)
            answerable = d.get("answerable", None)

            samples.append({
                "image": image_path,
                "question": question,
                "answers": answers,
                "answer_type": answer_type,
                "answerable": answerable,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample