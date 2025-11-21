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


def eval_vsr(results, annfile=None):
    with open(results, 'r') as f:
        res = json.load(f)
    preds = [pr for pred in res for pr in pred['predictions']]
    ans = [an for answer in res for an in answer['answers']]
    
    #NOTE used to be normalize_answer(p) == normalize_answer(a)
    is_correct = lambda a, p : (normalize_answer(a) == normalize_answer(p)) or (normalize_answer(a) in normalize_answer(p)) or (normalize_answer(p) in normalize_answer(a))
    
    correct = sum([is_correct(a, p) for p, a in zip(preds, ans)]) 
    total = len(ans)
    accuracy = correct / total * 100
    print(f"Overall Accuracy: {accuracy:.2f}")