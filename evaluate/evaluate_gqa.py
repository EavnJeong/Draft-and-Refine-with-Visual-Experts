import json
import os
import re
from tqdm import tqdm


def normalize_answer(s):
    """Lowercase, remove punctuation/articles/extra spaces."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def eval_gqa(resFile, annFile=None):
    """
    Evaluate GQA accuracy (Exact Match)

    Args:
        resFile (str): path to model predictions JSON file
                       format -> [{"question_id": "12345", "answer": "yes"}, ...]
    Returns:
        float: overall accuracy
    """

    with open(resFile, "r") as f:
        results = json.load(f)

    if isinstance(results, list):
        pred_dict = {str(r["question_id"]): r["answer"] for r in results}
    elif isinstance(results, dict):
        pred_dict = {str(k): v.get("answer", v) for k, v in results.items()}
    else:
        raise ValueError("Invalid results format.")

    with open(annFile, "r") as f:
        anns = json.load(f)

    total, correct = 0, 0
    mismatches = []

    for qid, ann in tqdm(anns.items()):
        gt = ann.get("answer", None)
        if gt is None or qid not in pred_dict:
            continue
        pred = pred_dict[qid]

        p, g = normalize_answer(pred), normalize_answer(gt)
        match = (p == g)
        correct += int(match)
        total += 1
        if not match:
            mismatches.append({"question_id": qid, "pred": pred, "gt": gt})

    acc = correct / total if total > 0 else 0
    print("=== GQA Evaluation (Accuracy Only) ===")
    print(f"Questions evaluated: {total}")
    print(f"Accuracy: {acc * 100:.2f}% ({correct}/{total})")
    print(f"Mismatched examples: {len(mismatches)}")
    return acc, mismatches
