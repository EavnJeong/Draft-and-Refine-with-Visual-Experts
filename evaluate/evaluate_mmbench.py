import json
from collections import defaultdict
from sklearn.metrics import accuracy_score
import re


def eval_mmbench(result_path, annfile=None):
    print("Evaluating MMBench (Legacy) ...")

    # --- Load results ---
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # --- Group by task (optional) ---
    grouped = defaultdict(list)
    for r in results:
        grouped[r.get("task", "mmbench").lower()].append(r)

    # --- Helper: normalize predictions ---
    def parse_pred(pred):
        """Extract the most meaningful short answer (A/B/C/D or text)."""
        p = str(pred).strip()
        if not p:
            return "NONE"

        # Normalize patterns like "Answer: A", "A.", "a)", "(A)"
        p = re.sub(r"^(Answer|Response|Choice|Option)[:：]?\s*", "", p, flags=re.IGNORECASE)
        p = re.sub(r"[\(\)\.\s]", "", p)
        p = p.strip()

        # Keep only A/B/C/D if detected
        if len(p) == 1 and p.upper() in ["A", "B", "C", "D"]:
            return p.upper()

        # Normalize yes/no forms
        if p.lower().startswith("yes"):
            return "YES"
        if p.lower().startswith("no"):
            return "NO"

        return p.upper()

    # --- Compute accuracy per task ---
    task_scores = {}
    total_preds, total_gts = [], []

    for task, recs in grouped.items():
        preds, gts = [], []

        for r in recs:
            pred = parse_pred(r["pred"])
            gt = str(r["gt"]).strip().upper()
            preds.append(pred)
            gts.append(gt)

        acc = accuracy_score(gts, preds)
        task_scores[task] = acc * 100
        total_preds.extend(preds)
        total_gts.extend(gts)

    overall_acc = accuracy_score(total_gts, total_preds) * 100

    # --- Print summary ---
    print("=" * 70)
    print(f"{'Task':<40} | {'Accuracy (%)':>12}")
    print("-" * 70)
    for t, s in task_scores.items():
        print(f"{t:<40} | {s:12.2f}")
    print("-" * 70)
    print(f"{'Overall Accuracy':<40} | {overall_acc:12.2f}")
    print("=" * 70)

    return {
        "overall_acc": overall_acc,
        "per_task": task_scores,
    }