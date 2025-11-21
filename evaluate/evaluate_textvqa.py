import json
import re


def normalize_answer(s):
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def vqa_accuracy(predictions, list_of_answers):
    """
    Args:
        predictions (list[str]): model predictions
        list_of_answers (list[list[str]]): list of 10 ground-truth answers per sample
    Returns:
        float: average VQA accuracy (%)
    """
    assert len(predictions) == len(list_of_answers)
    scores = []

    for pred, answers in zip(predictions, list_of_answers):
        pred_n = normalize_answer(pred)
        answers_n = [normalize_answer(a) for a in answers]

        # count matches by inclusion, not just equality
        count = 0
        for a in answers_n:
            if not a:
                continue
            # Either prediction includes the answer, or the answer includes the prediction
            if a in pred_n or pred_n in a:
                count += 1

        score = min(count / 3.0, 1.0)
        scores.append(score)

    return 100.0 * sum(scores) / len(scores)


def eval_textvqa(resFile, annfile=None):
    with open(resFile, 'r') as f:
        res = json.load(f)
    preds = [pr for pred in res for pr in pred['predictions']]
    ans = [an for answer in res for an in answer['answers']]
    accuracy = vqa_accuracy(preds, ans)
    print(f"Overall Accuracy: {accuracy:.2f}")