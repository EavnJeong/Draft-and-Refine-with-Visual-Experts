import os
from torch.utils.data import Dataset


class MMEAllDataset(Dataset):
    """
    Unified MME Dataset for all 14 tasks under one root directory.
    Supports both per-task subfolders and flat folder structures.
    Each .txt file may contain multiple (question, answer) pairs,
    formatted as:
        Does this artwork exist in the form of painting? Please answer yes or no.    Yes
        Does this artwork exist in the form of glassware? Please answer yes or no.   No
    """

    def __init__(self, root_dir, limit_per_task=19):
        self.root_dir = root_dir
        self.samples = []

        # scan all subfolders (tasks)
        task_list = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for task in task_list:
            task_dir = os.path.join(root_dir, task)

            # case A: images/ + questions_answers_YN/
            if os.path.exists(os.path.join(task_dir, "images")):
                img_dir = os.path.join(task_dir, "images")
                qa_dir = os.path.join(task_dir, "questions_answers_YN")
                img_names = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
                txt_names = {os.path.splitext(f)[0] for f in os.listdir(qa_dir)}
                common = sorted(img_names & txt_names)

                if limit_per_task is not None:
                    common = common[:limit_per_task]

                for fid in common:
                    img_path = os.path.join(img_dir, f"{fid}.jpg")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(img_dir, f"{fid}.png")
                    txt_path = os.path.join(qa_dir, f"{fid}.txt")
                    if not os.path.exists(txt_path):
                        continue

                    qa_pairs = self._parse_txt(txt_path)
                    for i, (q, a) in enumerate(qa_pairs):
                        self.samples.append({
                            "id": f"{fid}_{i}",
                            "image": img_path,
                            "question": q,
                            "answer": a,
                            "task": task
                        })
            # case B: mixed images + txt in same folder
            else:
                files = os.listdir(task_dir)
                img_names = {os.path.splitext(f)[0] for f in files if f.endswith((".jpg", ".png"))}
                txt_names = {os.path.splitext(f)[0] for f in files if f.endswith(".txt")}
                common = sorted(img_names & txt_names)

                if limit_per_task is not None:
                    common = common[:limit_per_task]

                for fid in common:
                    img_path = os.path.join(task_dir, f"{fid}.jpg")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(task_dir, f"{fid}.png")
                    txt_path = os.path.join(task_dir, f"{fid}.txt")
                    if not os.path.exists(txt_path):
                        continue

                    qa_pairs = self._parse_txt(txt_path)
                    for i, (q, a) in enumerate(qa_pairs):
                        self.samples.append({
                            "id": f"{fid}_{i}",
                            "image": img_path,
                            "question": q,
                            "answer": a,
                            "task": task
                        })

    def _parse_txt(self, txt_path):
        """
        Parse lines formatted as:
            Question text .......   Yes/No
        Returns a list of (question, answer) pairs.
        """
        qa_pairs = []
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for line in lines:
            # split from right side to keep last token (answer)
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                question, answer = parts
                qa_pairs.append((question.strip(), answer.strip()))
            else:
                qa_pairs.append((line.strip(), ""))

        return qa_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "image": s["image"],
            "question": s["question"],
            "answer": s["answer"],
            "task": s["task"],
            "id": s["id"]
        }