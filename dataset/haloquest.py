import os
import pandas as pd
import requests
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


class HaloQuestDataset(Dataset):
    """
    HaloQuest Dataset Loader (auto-download version)
    - Downloads all images from 'url' to root_dir/images/
    - Returns dict: {image_path, question, groundtruth, hallucination_type, image_type, question_id}
    """
    def __init__(
            self, 
            root_dir, 
            csv_name="haloquest-eval.csv", 
            limit=None, 
            verify_images=False, 
            download=True
        ):
        """
        Args:
            root_dir (str): Path containing the CSV file and images folder
            csv_name (str): CSV filename (default: 'HaloQuest.csv')
            limit (int or None): Number of samples per hallucination_type (None = use all)
            verify_images (bool): Skip missing local images if True
            download (bool): If True, download missing images from URLs
        """
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, csv_name)
        self.image_dir = os.path.join(root_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Missing file: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_cols = [
            "image_name", "url", "image type", "hallucination type",
            "question", "groundtruth responses", "split"
        ]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {self.csv_path}")

        # Download images if needed
        if download:
            print("[HaloQuest] Downloading missing images...")
            for _, row in tqdm(df.iterrows(), total=len(df)):
                img_name = row["image_name"]
                img_url = row["url"]
                save_path = os.path.join(self.image_dir, img_name)

                if not os.path.exists(save_path):
                    try:
                        r = requests.get(img_url, timeout=10)
                        if r.status_code == 200:
                            with open(save_path, "wb") as f:
                                f.write(r.content)
                        else:
                            print(f"⚠️ Failed to fetch {img_url} (status {r.status_code})")
                    except Exception as e:
                        print(f"⚠️ Error downloading {img_name}: {e}")

        # Group by hallucination type
        type_groups = defaultdict(list)
        for i, row in df.iterrows():
            img_name = row["image_name"]
            local_img = os.path.join(self.image_dir, img_name)
            if verify_images and not os.path.exists(local_img):
                continue

            sample = {
                "image_path": local_img,
                "image_url": row["url"],
                "image_type": row["image type"],
                "hallucination_type": row["hallucination type"],
                "question": str(row["question"]),
                "groundtruth": str(row["groundtruth responses"]),
                "split": str(row["split"]),
                "question_id": str(i),
            }
            type_groups[row["hallucination type"]].append(sample)

        # Apply per-type limit (if specified)
        samples = []
        for htype, items in type_groups.items():
            if limit is None:
                samples.extend(items)
            else:
                samples.extend(items[:limit])

        self.samples = samples
        print(f"[HaloQuest] Loaded {len(self.samples)} samples "
              f"({', '.join(f'{k}:{len(v)}' for k, v in type_groups.items())}) "
              f"from {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "image_path": item["image_path"],
            "question": item["question"],
            "groundtruth": item["groundtruth"],
            "hallucination_type": item["hallucination_type"],
            "image_type": item["image_type"],
            "split": item["split"],
            "question_id": item["question_id"],
        }