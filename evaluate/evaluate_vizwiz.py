import json
import os
from collections import Counter
from sklearn.metrics import precision_recall_curve, auc, f1_score

from evaluate.vizwiz.PythonHelperTools.vqaTools.vqa import VQA


def eval_vizwiz(resFile, annfile=None):
    with open(resFile, "r") as f:
        results = json.load(f)
    subset_imgs = {r["image"] for r in results}

    with open(annfile, "r") as f:
        anns = json.load(f)
    anns_subset = [a for a in anns if a["image"] in subset_imgs]

    tmp_ann = "/tmp/vizwiz_val_subset.json"
    with open(tmp_ann, "w") as f:
        json.dump(anns_subset, f)

    vqa = VQA(tmp_ann)
    vqaRes = VQA(resFile)

    imgs = vqa.getImgs()
    total_acc = 0.0
    total_count = 0
    per_type_acc = {}
    per_type_count = {}
    y_true, y_pred = [], []

    for img in imgs:
        if img not in vqaRes.imgToQA:
            continue

        gt_ann = vqa.getAnns(img)[0]
        pred_ann = vqaRes.getAnns(img)[0]

        gt_answers = [a["answer"].strip().lower() for a in gt_ann["answers"]]
        pred_answer = pred_ann["answer"].strip().lower()
        ans_type = gt_ann.get("answer_type", "other")

        acc = min(Counter(gt_answers)[pred_answer] / 3, 1.0)
        total_acc += acc
        total_count += 1

        per_type_acc[ans_type] = per_type_acc.get(ans_type, 0) + acc
        per_type_count[ans_type] = per_type_count.get(ans_type, 0) + 1

        gt_ansb = gt_ann.get("answerable", 1)
        pred_ansb = 0 if pred_answer == "unanswerable" else 1
        y_true.append(gt_ansb)
        y_pred.append(pred_ansb)

    overall_acc = total_acc / total_count if total_count else 0
    per_type = {t: per_type_acc[t] / per_type_count[t] for t in per_type_acc}

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = auc(recall, precision)
    f1 = f1_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])

    print("=== VizWiz Evaluation (Subset Mode) ===")
    print(f"Images evaluated: {total_count}")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"Unanswerability - AP: {ap:.3f}, F1: {f1:.3f}\n")
    print("Per Answer Type Accuracy:")
    for t, v in per_type.items():
        print(f"  {t:15s}: {v * 100:.2f}%")
    os.remove(tmp_ann)