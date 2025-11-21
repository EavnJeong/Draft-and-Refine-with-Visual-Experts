import json
import os
import tempfile
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def evaluate_nocaps(results_or_path, gt_file=None):
    """
    Evaluate NoCaps predictions (CIDEr only).
    Accepts:
        - results (list of dict)
        - or path to a JSON file

    Automatically maps OpenImages-style image_ids to COCO-style integer IDs.
    """

    if isinstance(results_or_path, list):
        preds = results_or_path
    elif isinstance(results_or_path, str):
        with open(results_or_path, "r") as f:
            preds = json.load(f)
    else:
        raise TypeError(f"Unsupported input type: {type(results_or_path)}")

    print(f"Loading GT: {gt_file}")
    coco = COCO(gt_file)
    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    id_map = {img["open_images_id"]: img["id"] for img in gt_data["images"]}
    print(f"Loaded {len(id_map)} ID mappings from GT.")

    fixed_preds = []
    unmapped = 0
    for p in preds:
        pid = str(p.get("image_id", p.get("id", "")))
        if pid in id_map:
            fixed_preds.append({"image_id": id_map[pid], "caption": p.get("final_caption", p.get("caption", ""))})
        elif pid.isdigit():
            fixed_preds.append({"image_id": int(pid), "caption": p.get("final_caption", p.get("caption", ""))})
        else:
            unmapped += 1
    if unmapped:
        print(f"[WARN] {unmapped} predictions could not be mapped to GT IDs and were skipped.")
    if not fixed_preds:
        raise ValueError("No valid image_ids found after mapping.")

    tmp_dir = tempfile.mkdtemp()
    pred_path = os.path.join(tmp_dir, "nocaps_preds_fixed.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(fixed_preds, f, indent=2, ensure_ascii=False)

    cocoRes = coco.loadRes(pred_path)
    domain_map = {img["id"]: img.get("domain", "unknown") for img in gt_data["images"]}

    def _eval_subset(image_ids, label):
        if not image_ids:
            return 0.0
        coco_subset = coco
        cocoRes_subset = cocoRes
        subset_ann = [ann for ann in coco.dataset["annotations"] if ann["image_id"] in image_ids]
        if not subset_ann:
            return 0.0
        coco_subset.dataset["annotations"] = subset_ann
        coco_subset.dataset["images"] = [img for img in coco.dataset["images"] if img["id"] in image_ids]
        coco_subset.createIndex()

        cocoEval = COCOEvalCap(coco_subset, cocoRes_subset)
        cocoEval.params["image_id"] = image_ids
        cocoEval.evaluate()
        return cocoEval.eval.get("CIDEr", 0.0)

    valid_ids = [p["image_id"] for p in fixed_preds]
    cider_all = _eval_subset(valid_ids, "overall")
    in_ids = [i for i in valid_ids if domain_map.get(i) == "in-domain"]
    near_ids = [i for i in valid_ids if domain_map.get(i) == "near-domain"]
    out_ids = [i for i in valid_ids if domain_map.get(i) == "out-domain"]

    cider_in = _eval_subset(in_ids, "in-domain")
    cider_near = _eval_subset(near_ids, "near-domain")
    cider_out = _eval_subset(out_ids, "out-domain")

    print("\n========== NoCaps CIDEr Evaluation ==========")
    print(f"Overall CIDEr       : {cider_all:.4f}")
    print(f"In-domain CIDEr     : {cider_in:.4f}")
    print(f"Near-domain CIDEr   : {cider_near:.4f}")
    print(f"Out-of-domain CIDEr : {cider_out:.4f}")
    print("=============================================\n")

    return {
        "CIDEr_all": cider_all,
        "CIDEr_in": cider_in,
        "CIDEr_near": cider_near,
        "CIDEr_out": cider_out,
    }