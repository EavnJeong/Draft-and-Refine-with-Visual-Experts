import json
from collections import defaultdict
from sklearn.metrics import accuracy_score


def eval_mme(result_path, annfile=None):
    print("Evaluating MME Benchmark ...")

    # --- Load results ---
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Group results by task
    grouped = defaultdict(list)
    for r in results:
        grouped[r["task"].lower()].append(r)

    # MME benchmark category structure
    eval_type_dict = {
        "Perception": [
            "existence", "count", "position", "color",
            "posters", "celebrity", "scene", "landmark", "artwork", "ocr"
        ],
        "Cognition": [
            "commonsense_reasoning", "numerical_calculation",
            "text_translation", "code_reasoning"
        ],
    }

    # Normalize predictions to yes/no/other
    def parse_pred_ans(pred_ans: str):
        p = str(pred_ans).strip().lower()
        if p in ["yes", "no"]:
            return p
        if p.startswith("yes"):
            return "yes"
        if p.startswith("no"):
            return "no"
        if any(x in p for x in ["yeah", "yep", "sure"]):
            return "yes"
        if any(x in p for x in ["nah", "nope", "unanswerable", "not really"]):
            return "no"
        return "other"

    # Compute accuracy and acc+ (pairwise)
    def compute_metrics(gts, preds):
        label_map = {"yes": 1, "no": 0, "other": -1}
        gts = [label_map[g] for g in gts]
        preds = [label_map[p] for p in preds]

        acc = accuracy_score(gts, preds)
        clean_gts, clean_preds = [], []
        for g, p in zip(gts, preds):
            if p != -1:
                clean_gts.append(g)
                clean_preds.append(p)
        acc_clean = accuracy_score(clean_gts, clean_preds) if clean_gts else 0.0
        return acc, acc_clean

    # --- Evaluate ---
    final_stats = {}
    total_score = 0.0
    perception_score = 0.0
    cognition_score = 0.0

    for eval_type, task_list in eval_type_dict.items():
        print(f"\n========== {eval_type} ==========")
        eval_score = 0.0
        for task in task_list:
            if task not in grouped:
                continue

            recs = grouped[task]
            # group by base_id (ex: "0001_0" and "0001_1" → "0001")
            img_groups = defaultdict(list)
            for r in recs:
                base_id = r["id"].rsplit("_", 1)[0]
                img_groups[base_id].append(r)

            gts, preds = [], []
            acc_plus_correct = 0

            for img_id, items in img_groups.items():
                if len(items) != 2:
                    continue
                img_correct = 0
                for it in items:
                    gt = str(it["gt"]).strip().lower()
                    pred = parse_pred_ans(it["pred"])
                    gts.append(gt)
                    preds.append(pred)
                    if gt == pred:
                        img_correct += 1
                if img_correct == 2:
                    acc_plus_correct += 1

            acc, _ = compute_metrics(gts, preds)
            img_num = len(img_groups)
            acc_plus = acc_plus_correct / img_num if img_num > 0 else 0.0
            task_score = (acc + acc_plus) * 100.0
            eval_score += task_score

            print(f"[{task:<25}] Score: {task_score:6.2f}  (Acc={acc*100:.2f}%, Acc+={acc_plus*100:.2f}%)")

        print(f"{eval_type} subtotal: {eval_score:.2f}\n")
        final_stats[eval_type] = eval_score
        total_score += eval_score

        if eval_type == "Perception":
            perception_score = eval_score
        else:
            cognition_score = eval_score

    # --- Normalize to official scales ---
    perception_scaled = perception_score / (len(eval_type_dict["Perception"]) * 200) * 2000
    cognition_scaled = cognition_score / (len(eval_type_dict["Cognition"]) * 200) * 800
    total_scaled = perception_scaled + cognition_scaled

    print("=" * 70)
    print(f"Perception (A): {perception_scaled:.2f} / 2000")
    print(f"Cognition  (B): {cognition_scaled:.2f} / 800")
    print("-" * 70)
    print(f"Overall MME Score: {total_scaled:.2f} / 2800.00")
    print("=" * 70)

    return {
        "Perception": perception_scaled,
        "Cognition": cognition_scaled,
        "Overall": total_scaled,
    }