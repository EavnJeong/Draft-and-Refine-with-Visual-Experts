import json
import os
from torch.utils.data import Dataset



class VCRDataset(Dataset):
    def __init__(self, root_dir, split="val", limit=None):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.root_dir = root_dir
        self.image_root = os.path.join(root_dir, "vcr1images")
        self.anno_path = os.path.join(root_dir, "anno", f"{split}.jsonl")

        assert os.path.exists(self.anno_path), f"Annotation file not found: {self.anno_path}"
        with open(self.anno_path, "r") as f:
            self.items = [json.loads(line) for line in f]

        if limit is not None:
            self.items = self.items[:limit]

        print(f"[SimpleVCRDataset] Loaded {len(self.items)} samples from {self.anno_path}")

    def __len__(self):
        return len(self.items)

    def _tokens_to_text(self, tokens):
        """Flatten VCR token lists and placeholders."""
        flat = []
        for tok in tokens:
            if isinstance(tok, list):
                flat.append(f"[obj{tok[0]}]")
            else:
                flat.append(tok)
        return " ".join(flat).replace(" ,", ",").replace(" .", ".")

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.image_root, item["img_fn"])

        objects = item.get("objects", [])

        def resolve_tokens(tokens):
            """Convert list tokens like [2] into readable '[object_name_index]' placeholders."""
            flat = []
            for tok in tokens:
                if isinstance(tok, list):
                    for i in tok:
                        if 0 <= i < len(objects):
                            obj_name = objects[i]
                            flat.append(f"[{obj_name}{i}]")
                        else:
                            flat.append(f"[obj{i}]")
                else:
                    flat.append(tok)
            return (
                " ".join(flat)
                .replace(" ,", ",")
                .replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
            )
        q_text = resolve_tokens(item["question"])
        ans_choices = [resolve_tokens(a) for a in item["answer_choices"]]
        rat_choices = [resolve_tokens(r) for r in item.get("rationale_choices", [])]

        ans_label = item.get("answer_label", None)
        rat_label = item.get("rationale_label", None)

        gold_answer = (
            ans_choices[ans_label]
            if ans_label is not None and 0 <= ans_label < len(ans_choices)
            else None
        )
        gold_rationale = (
            rat_choices[rat_label]
            if rat_label is not None and 0 <= rat_label < len(rat_choices)
            else None
        )
        return {
            "image": img_path,               # absolute image path
            "question": q_text,              # resolved question text
            "answer_choice": ans_choices,          # list[str] answer candidates
            "rationales": rat_choices,       # list[str] rationale candidates
            "answers_texts": gold_answer,    # ground-truth answer string
            "rationale_texts": gold_rationale, # ground-truth rationale string
            "answer_label": ans_label+1,        # ground-truth answer index
            "rationale_label": rat_label+1,     # ground-truth rationale index
            "question_ids": item.get("annot_id", f"vcr_{idx}"),  # ID for eval
        }