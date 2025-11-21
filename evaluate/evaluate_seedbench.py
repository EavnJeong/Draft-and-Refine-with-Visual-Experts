import json
from collections import defaultdict
from sklearn.metrics import accuracy_score


def eval_seedbench(result_path, annfile=None):
    """
    Evaluation for SEED-Bench (Multiple-Choice)
    - Computes overall accuracy and per-question-type accuracy
    - Results file must contain fields: {question_id, question_type, pred, gt}
    """
    print("Evaluating SEED-Bench ...")

    # --- Load results ---
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # --- Group results by question_type ---
    grouped = defaultdict(list)
    for r in results:
        qtype = str(r["question_type"]).strip()
        grouped[qtype].append(r)

    # --- Compute accuracies ---
    overall_preds, overall_gts = [], []
    per_type_scores = {}

    for qtype, recs in grouped.items():
        preds = [str(r["pred"]).strip().upper() for r in recs]
        gts = [str(r["gt"]).strip().upper() for r in recs]

        acc = accuracy_score(gts, preds)
        per_type_scores[qtype] = {
            "acc": acc * 100.0,
            "count": len(recs)
        }

        overall_preds.extend(preds)
        overall_gts.extend(gts)

    # --- Overall Accuracy ---
    overall_acc = accuracy_score(overall_gts, overall_preds) * 100.0

    # --- Print detailed results ---
    print("\n========== SEED-Bench Accuracy ==========")
    for qtype, stats in sorted(per_type_scores.items(), key=lambda x: -x[1]["acc"]):
        print(f"[{qtype:<25}]  Acc: {stats['acc']:6.2f}%   (N={stats['count']})")
    print("-" * 60)
    print(f"Overall Accuracy: {overall_acc:6.2f}%")
    print("=" * 60)

    return {
        "Overall Accuracy": overall_acc,
        "Per-Type Accuracy": per_type_scores,
    }
