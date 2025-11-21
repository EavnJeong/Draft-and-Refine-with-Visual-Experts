import os
import json
import requests
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


class MMHalBenchDataset(Dataset):
    """
    MMHal-Bench Dataset Loader (auto-download version)
    - Downloads all images from 'image_src' to root_dir/images/
    - Returns dict: {image_path, question, gt_answer, question_type, question_topic, image_content, image_id}
    """

    def __init__(
            self, 
            root_dir, 
            json_name="response_template.json", 
            limit=None, 
            verify_images=False, 
            download=True
        ):
        """
        Args:
            root_dir (str): Path containing the JSON file and images folder
            json_name (str): JSON filename (default: 'response_template.json')
            limit (int or None): Number of samples per question_type (None = use all)
            verify_images (bool): Skip missing local images if True
            download (bool): If True, download missing images from URLs
        """
        self.root_dir = root_dir
        self.json_path = os.path.join(root_dir, json_name)
        self.image_dir = os.path.join(root_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Missing file: {self.json_path}")

        with open(self.json_path, "r") as f:
            data = json.load(f)

        required_keys = [
            "image_id", "image_src", "question", "gt_answer",
            "question_type", "question_topic", "image_content"
        ]
        for k in required_keys:
            if not all(k in d for d in data):
                raise ValueError(f"Missing required key '{k}' in some JSON entries")

        # --- Download missing images ---
        if download:
            print("[MMHalBench] Checking & downloading missing images...")
            for d in tqdm(data, total=len(data)):
                img_url = d["image_src"]
                filename = os.path.basename(img_url)
                save_path = os.path.join(self.image_dir, filename)

                # Skip if already exists
                if os.path.exists(save_path):
                    continue

                try:
                    r = requests.get(img_url, timeout=10)
                    if r.status_code == 200:
                        with open(save_path, "wb") as f:
                            f.write(r.content)
                    else:
                        print(f"⚠️ Failed to fetch {img_url} (status {r.status_code})")
                except Exception as e:
                    print(f"⚠️ Error downloading {img_url}: {e}")

        # --- Group by question type ---
        type_groups = defaultdict(list)
        for i, d in enumerate(data):
            filename = os.path.basename(d["image_src"])
            local_img = os.path.join(self.image_dir, filename)
            if verify_images and not os.path.exists(local_img):
                continue

            sample = {
                "image_path": local_img,
                "image_url": d["image_src"],
                "image_id": d["image_id"],
                "image_content": d["image_content"],
                "question": d["question"],
                "gt_answer": d["gt_answer"],
                "question_type": d["question_type"],
                "question_topic": d["question_topic"],
                "question_id": str(i),
            }
            type_groups[d["question_type"]].append(sample)

        # --- Apply per-type limit ---
        samples = []
        for qtype, items in type_groups.items():
            if limit is None:
                samples.extend(items)
            else:
                samples.extend(items[:limit])

        self.samples = samples
        print(f"[MMHalBench] Loaded {len(self.samples)} samples "
              f"({', '.join(f'{k}:{len(v)}' for k, v in type_groups.items())}) "
              f"from {self.json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "image_path": item["image_path"],
            "question": item["question"],
            "gt_answer": item["gt_answer"],
            "question_type": item["question_type"],
            "question_topic": item["question_topic"],
            "image_content": item["image_content"],
            "image_id": item["image_id"],
            "question_id": item["question_id"],
        }