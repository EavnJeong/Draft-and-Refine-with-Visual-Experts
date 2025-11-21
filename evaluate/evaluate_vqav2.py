import json
import os
import tempfile
from collections import Counter

from evaluate.VQA.PythonHelperTools.vqaTools.vqa import VQA
from evaluate.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


def eval_vqav2(results_file, annfile=None):
    annFile  = "evaluate/VQA/v2_mscoco_val2014_annotations.json"
    quesFile = "evaluate/VQA/v2_OpenEnded_mscoco_val2014_questions.json"

    with open(results_file, "r") as f:
        results = json.load(f)
    subset_qids = {r["question_id"] for r in results}

    with open(annFile, "r") as f:
        anns = json.load(f)
    anns["annotations"] = [a for a in anns["annotations"] if a["question_id"] in subset_qids]

    with open(quesFile, "r") as f:
        ques = json.load(f)
    ques["questions"] = [q for q in ques["questions"] if q["question_id"] in subset_qids]

    tmp_ann = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_ques = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    json.dump(anns, open(tmp_ann.name, "w"))
    json.dump(ques, open(tmp_ques.name, "w"))

    vqa = VQA(tmp_ann.name, tmp_ques.name)
    vqaRes = vqa.loadRes(results_file, tmp_ques.name)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()

    print("Overall Accuracy: %.02f" % vqaEval.accuracy['overall'])
    print("Per Question Type Accuracy:")
    for k, v in vqaEval.accuracy['perQuestionType'].items():
        print(f"{k}: {v:.02f}")
    print("Per Answer Type Accuracy:")
    for k, v in vqaEval.accuracy['perAnswerType'].items():
        print(f"{k}: {v:.02f}")

    os.unlink(tmp_ann.name)
    os.unlink(tmp_ques.name)