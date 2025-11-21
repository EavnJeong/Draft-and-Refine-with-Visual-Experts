import json


def eval_vcr(results_path, annfile=None):
    """
    Evaluate VCR results for:
      - Q→A  : only answer is correct
      - Q→R  : only rationale is correct
      - Q→AR : both answer and rationale are correct

    Expected JSON structure:
    [
      {
        "pred_answer": [...],
        "pred_rationale": [...],
        "answers": [...],
        "rationale_labels": [...]
      },
      ...
    ]
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    total = 0
    correct_a = 0   # Q→A
    correct_r = 0   # Q→R
    correct_ar = 0  # Q→AR

    for entry in results:
        pred_a = entry.get("pred_answer", [])
        pred_r = entry.get("pred_rationale", [])
        gt_a = entry.get("answers", [])
        gt_r = entry.get("rationale_labels", [])

        for pa, pr, ga, gr in zip(pred_a, pred_r, gt_a, gt_r):
            total += 1
            pa, pr, ga, gr = int(pa), int(pr), int(ga), int(gr)

            a_match = pa == ga
            r_match = pr == gr

            if a_match:
                correct_a += 1  # answer only
            if r_match:
                correct_r += 1  # rationale only
            if a_match and r_match:
                correct_ar += 1  # both correct

    qa_acc = 100.0 * correct_a / total if total > 0 else 0.0
    qr_acc = 100.0 * correct_r / total if total > 0 else 0.0
    qar_acc = 100.0 * correct_ar / total if total > 0 else 0.0

    print("===== VCR Evaluation =====")
    print(f"Q→A  (Answer only)    : {qa_acc:.2f}%  ({correct_a}/{total})")
    print(f"Q→R  (Rationale only) : {qr_acc:.2f}%  ({correct_r}/{total})")
    print(f"Q→AR (Both correct)   : {qar_acc:.2f}%  ({correct_ar}/{total})")