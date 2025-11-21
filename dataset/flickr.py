import os
from torch.utils.data import Dataset
from PIL import Image


class FlickrDataset(Dataset):
    """
    Flickr Captioning Dataset Loader
    - Parses results.csv where each line = "image_name.jpg| idx | caption"
    - Splits into train/val using val.txt IDs (without .jpg)
    - Returns dict with {"image": image_path, "caption": caption}
    """

    def __init__(self, root_dir, split="train", limit=None):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")

        results_path = os.path.join(root_dir, "results.csv")
        val_path = os.path.join(root_dir, "val.txt")

        # --- Load validation IDs ---
        with open(val_path, "r") as f:
            val_ids = set(line.strip() for line in f if line.strip())

        # --- Parse results.csv ---
        samples = []
        with open(results_path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 3:
                    continue
                filename = parts[0].strip()
                caption = parts[2].strip()

                img_id = os.path.splitext(filename)[0]
                if split == "val" and img_id in val_ids:
                    samples.append((filename, caption))
                elif split == "train" and img_id not in val_ids:
                    samples.append((filename, caption))

        self.samples = samples
        if limit is not None:
            self.samples = self.samples[:limit]
        print(f"[FlickrDataset] Loaded {len(self.samples)} samples for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        return {
            "image": image_path,
            "caption": caption,
        }
