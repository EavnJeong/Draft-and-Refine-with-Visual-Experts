import json
import os
import tempfile
from evaluate.VQA.PythonHelperTools.vqaTools.vqa import VQA
from evaluate.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


def eval_okvqa(results_file, annfile=None):
    """
    Evaluate OK-VQA results using the official VQA evaluation protocol.
    Requires the same folder structure as your eval_vqav2 version.

    Args:
        results_file (str): path to model predictions JSON file
                            [{"question_id": 123, "answer": "dog"}, ...]

    Example:
        eval_okvqa("results/okvqa_pred.json")
    """
    annFile, quesFile = annfile.split("|")

    # --- Filter only question_ids included in the results ---
    with open(results_file, "r") as f:
        results = json.load(f)
    subset_qids = {r["question_id"] for r in results}

    with open(annFile, "r") as f:
        anns = json.load(f)
    anns["annotations"] = [a for a in anns["annotations"] if a["question_id"] in subset_qids]

    with open(quesFile, "r") as f:
        ques = json.load(f)
    ques["questions"] = [q for q in ques["questions"] if q["question_id"] in subset_qids]

    # --- Create temporary filtered JSON files ---
    tmp_ann = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_ques = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    json.dump(anns, open(tmp_ann.name, "w"))
    json.dump(ques, open(tmp_ques.name, "w"))

    # --- Run VQA evaluation ---
    vqa = VQA(tmp_ann.name, tmp_ques.name)
    vqaRes = vqa.loadRes(results_file, tmp_ques.name)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()

    # --- Print results ---
    print("=== OK-VQA Evaluation ===")
    print(f"Overall Accuracy: {vqaEval.accuracy['overall']:.02f}")
    print("\nPer Question Type Accuracy:")
    for k, v in vqaEval.accuracy['perQuestionType'].items():
        print(f"  {k}: {v:.02f}")
    print("\nPer Answer Type Accuracy:")
    for k, v in vqaEval.accuracy['perAnswerType'].items():
        print(f"  {k}: {v:.02f}")

    # --- Clean up temp files ---
    os.unlink(tmp_ann.name)
    os.unlink(tmp_ques.name)
