import os, io, base64
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MMBenchDataset(Dataset):
    """
    MMBench (Legacy) Dataset Loader
    - Returns clean question (no appended choices)
    - Separate 'choices' list and keeps 'hint'/'comment'
    """

    def __init__(self, root_dir, limit_per_task=19):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.samples = []

        legacy_path = os.path.join(root_dir, "MMBench_DEV_EN_legacy.tsv")
        if not os.path.exists(legacy_path):
            raise FileNotFoundError(f"Missing file: {legacy_path}")

        df = pd.read_csv(legacy_path, sep="\t")
        print(f"[Loaded TSV] {legacy_path} — {len(df)} rows")

        if "image" not in df.columns:
            raise ValueError("This TSV has no 'image' column — you need the Legacy version with Base64 data.")

        # ---- Group by category ----
        if "category" in df.columns:
            grouped = df.groupby("category")
        elif "l2-category" in df.columns:
            grouped = df.groupby("l2-category")
        else:
            grouped = [("global", df)]

        for subtask, subdf in grouped:
            # Apply limit only if limit_per_task is not None
            if limit_per_task is not None:
                subdf = subdf.head(limit_per_task)

            for i, row in subdf.iterrows():
                sid = f"MMBench_DEV_EN_{subtask}_{i}"
                img_path = os.path.join(self.image_dir, f"{sid}.jpg")

                # Decode image if needed
                if not os.path.exists(img_path):
                    try:
                        img_data = base64.b64decode(
                            row["image"].split(",")[-1] if "," in row["image"] else row["image"]
                        )
                        Image.open(io.BytesIO(img_data)).convert("RGB").save(img_path)
                    except Exception as e:
                        print(f"[Decode Error @ {sid}]: {e}")
                        Image.new("RGB", (224, 224), (128, 128, 128)).save(img_path)

                # Only question
                q_text = row.get("question", "").strip()

                # Separate choices
                choices = []
                for opt_label in ["A", "B", "C", "D"]:
                    val = row.get(opt_label, None)
                    if isinstance(val, str) and val.strip() and val.lower() != "nan":
                        choices.append(f"{opt_label}: {val.strip()}")

                hint = row.get("hint", "")
                comment = row.get("comment", "")

                self.samples.append({
                    "id": sid,
                    "image_path": img_path,
                    "question": q_text, 
                    "choices": choices, 
                    "answer": row.get("answer", ""),
                    "hint": hint if isinstance(hint, str) else "",
                    "comment": comment if isinstance(comment, str) else "",
                    "task": f"MMBench_DEV_EN:{subtask}",
                })
        print(f"[Decoded] {len(self.samples)} samples saved to {self.image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]