import os
import json
from torch.utils.data import Dataset
from collections import defaultdict


class NoCapsDataset(Dataset):
    """
    Inference-time NoCaps dataset: one entry per image.
    """
    def __init__(self, root_dir, limit=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.annotation_path = os.path.join(root_dir, "nocap_val_4500_captions.json")

        with open(self.annotation_path, "r") as f:
            data = json.load(f)

        # id -> info
        id_to_info = {
            img["id"]: {
                "open_images_id": img["open_images_id"],
                "file_name": img["file_name"],
                "domain": img["domain"]
            }
            for img in data["images"]
        }

        # image_id -> list of 5 captions
        refs_dict = defaultdict(list)
        for ann in data["annotations"]:
            refs_dict[ann["image_id"]].append(ann["caption"])

        self.samples = []
        for img_id, info in id_to_info.items():
            image_path = os.path.join(self.image_dir, info["file_name"])
            refs = refs_dict[img_id]
            self.samples.append({
                "image": image_path,
                "image_id": info["open_images_id"],
                "captions": refs,
                "domain": info["domain"],
            })
        if limit is not None:
            self.samples = self.samples[:limit]

        print(f"Loaded {len(self.samples)} images for inference (each with 5 GT captions).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image": sample["image"],
            "captions": sample["captions"],  # list of 5 GT captions
            "image_id": sample["image_id"],
            "domain": sample["domain"]
        }