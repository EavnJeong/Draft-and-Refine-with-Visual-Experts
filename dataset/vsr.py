import os
from torch.utils.data import Dataset

from datasets import load_dataset
import requests
import tqdm


class VSRDataset(Dataset):
    def __init__(self, root_dir, split="test", limit=None):
        self.split = split
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, "images")
        suffix = f"[:{limit}]" if limit is not None else ""
        self.dataset = load_dataset("cambridgeltl/vsr_random", cache_dir=self.root_dir, split=split+suffix)
        
        self._validate_images()
        
    def _validate_images(self):
        
        for data in tqdm.tqdm(self.dataset, desc="Validating if images are downloaded"):
            fname = data["image"]
            img_path = os.path.join(self.img_dir, fname)
            if not os.path.exists(img_path):
                try:
                    # image field may be either URL or HF image object
                    url = data["image_link"]

                    # download if it's a URL string
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    with open(img_path, "wb") as f:
                        f.write(r.content)

                except Exception as e:
                    print(f"Failed to download {fname}: {e}")
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        caption = item["caption"]
        question = f"Does the image show the following caption as true? {caption}"
        answer = "Yes" if item["label"] else "No"
        image_fname = item["image"]
        image_path = os.path.join(self.img_dir, image_fname)

        return {
            "image": image_path,
            "question": question,
            "answer": answer,
        }