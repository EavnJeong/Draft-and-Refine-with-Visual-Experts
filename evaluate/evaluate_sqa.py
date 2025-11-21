import json


def eval_sqa(results, annfile=None):
    with open(results, 'r') as f:
        res = json.load(f)
    preds = [pr for pred in res for pr in pred['predictions']]
    ans = [an for answer in res for an in answer['answers']]

    correct = sum([p==a for p, a in zip(preds, ans)])
    total = len(ans)
    accuracy = correct / total * 100
    print(f"Overall Accuracy: {accuracy:.2f}")