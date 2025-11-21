from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import json, tempfile, os


def evaluate_cococaption(results, gt_file=None):
    """
    Evaluate COCO-style captioning results using pycocoevalcap metrics.
    Fixes KeyError: 'info' by ensuring full COCO annotation structure.
    """
    annotations, predictions = [], []
    for idx, item in enumerate(results):
        image_ids = item["image"]
        answers = item.get("final_caption", item.get("answer"))
        refs = item.get("captions_gt", item.get("captions"))

        # Normalize types
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        if isinstance(answers, str):
            answers = [answers]
        if isinstance(refs, str):
            refs = [refs]

        for i, (img_path, pred, ref) in enumerate(zip(image_ids, answers, refs)):
            image_id = idx * 1000 + i
            annotations.append({
                "image_id": image_id,
                "id": image_id,
                "caption": ref
            })
            predictions.append({
                "image_id": image_id,
                "caption": pred
            })

    tmp_dir = tempfile.mkdtemp()
    ref_path = os.path.join(tmp_dir, "refs.json")
    pred_path = os.path.join(tmp_dir, "preds.json")

    coco_data = {
        "info": {"description": "COCO-style dummy caption annotations"},
        "images": [{"id": ann["image_id"]} for ann in annotations],
        "annotations": annotations,
        "licenses": [],
    }

    with open(ref_path, "w") as f:
        json.dump(coco_data, f)
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    coco = COCO(ref_path)
    cocoRes = coco.loadRes(pred_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    scores = {metric: score for metric, score in cocoEval.eval.items()}

    print("\n===== COCO Caption Evaluation =====")
    for k, v in scores.items():
        print(f"{k:<10}: {v:.4f}")

    return scores