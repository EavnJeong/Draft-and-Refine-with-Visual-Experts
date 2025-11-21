import json
import string
from tqdm import tqdm


def normalize_answer(s: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and trimming whitespace."""
    return s.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def eval_aokvqa(
    results_file: str,
    ann_path: str,
):
    """
    Evaluate A-OKVQA results using the official AllenAI protocol.
    - Matches predictions with ground truth question_id (string-based)
    - Computes exact-match accuracy
    - Reports both overall and per-question-type accuracy

    Args:
        results_file (str): Path to model predictions JSON file.
            Expected format:
                [{"question_id": "22jbM6gDxdaMaunuzgrsBB", "answer": "cigarette"}, ...]
        ann_path (str): Path to A-OKVQA annotation JSON (val/test split)
    """

    # --- Load predictions ---
    with open(results_file, "r") as f:
        preds_raw = json.load(f)
    preds = {
        str(r["question_id"]): normalize_answer(r["answer"])
        for r in preds_raw
        if r.get("answer", "").strip() != ""
    }

    # --- Load ground truth ---
    with open(ann_path, "r") as f:
        gt_data = json.load(f)

    total, correct = 0, 0
    per_type = {}

    for item in tqdm(gt_data, desc="Evaluating A-OKVQA"):
        qid = str(item["question_id"])
        gt_answers = [normalize_answer(a) for a in item.get("direct_answers", [])]
        q_type = item.get("question_type", "default")

        if qid not in preds:
            continue

        total += 1
        pred = preds[qid]

        # Exact-match check
        if pred in gt_answers:
            correct += 1
            per_type.setdefault(q_type, [0, 0])[0] += 1 
        per_type.setdefault(q_type, [0, 0])[1] += 1

    acc = (correct / total * 100) if total > 0 else 0.0

    # --- Print results ---
    print("\n=== A-OKVQA Evaluation ===")
    print(f"Overall Accuracy: {acc:.2f}% ({correct}/{total})")
    print("\nPer Question Type Accuracy:")
    for qtype, (c, t) in per_type.items():
        print(f"  {qtype}: {c / t * 100:.2f}% ({c}/{t})")

    return {
        "overall": acc,
        "per_type": {k: v[0] / v[1] * 100 for k, v in per_type.items()},
    }