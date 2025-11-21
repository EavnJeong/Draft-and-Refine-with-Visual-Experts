import json
import os
from torch.utils.data import Dataset


class OKVQADataset(Dataset):
    """
    Dataset class for OK-VQA (same format as VQAv2 / OK-VQA)
    Directory structure:
        okvqa/
        ├── images -> /media/Data1/qa/vqav2/images/
        ├── OpenEnded_mscoco_val2014_questions.json
        └── mscoco_val2014_annotations.json
    """

    def __init__(self, root_dir, split="val", limit=None):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.root_dir = root_dir
        self.split = split

        # image folder (symlink to COCO)
        self.image_dir = os.path.join(root_dir, "images", f"{split}2014")

        # file paths
        question_file = f"OpenEnded_mscoco_{split}2014_questions.json"
        annotation_file = f"mscoco_{split}2014_annotations.json"

        question_path = os.path.join(root_dir, question_file)
        annotation_path = os.path.join(root_dir, annotation_file)

        # load JSONs
        with open(question_path, "r") as f:
            q_data = json.load(f)["questions"]

        with open(annotation_path, "r") as f:
            a_data = json.load(f)["annotations"]

        # map annotation by question_id for fast lookup
        ann_map = {a["question_id"]: a for a in a_data}

        # combine question + answer
        self.samples = []
        for q in q_data:
            qid = q["question_id"]
            image_id = q["image_id"]
            image_path = os.path.join(self.image_dir, f"COCO_{split}2014_{image_id:012d}.jpg")

            ann = ann_map.get(qid, {})
            answers = [a["answer"] for a in ann.get("answers", [])] if "answers" in ann else []

            self.samples.append({
                "question_id": qid,
                "image_id": image_id,
                "image": image_path,
                "question": q["question"],
                "answers": answers,
                "multiple_choice_answer": ann.get("multiple_choice_answer", None),
            })

        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
