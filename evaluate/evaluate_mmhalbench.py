import os
import json
from openai import OpenAI
import base64
from tqdm import tqdm
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def encode_image_base64(image_path):
    """Convert local image file to base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def eval_mmhalbench(
    result_path,
    ann_file=None,
    json_gt="response_template.json",
    save_path="mmhal_eval_results.json"
):
    """
    Evaluate MMHal-Bench results using GPT-4V as evaluator.

    GPT-4V judges each sample as one of:
      ["correct", "grounded", "misperception", "hallucination"]
    """

    # --- Load model outputs ---
    with open(result_path, "r") as f:
        preds = json.load(f)
    if isinstance(preds, dict):
        preds = preds.get("results", preds)

    # --- Load ground truth ---
    if os.path.exists(json_gt):
        with open(json_gt, "r") as f:
            gt_all = {x["question"]: x for x in json.load(f)}
    else:
        gt_all = {x["question"]: x for x in preds}

    results = []
    print(f"[Evaluator] Using GPT-4V to assess {len(preds)} samples...")

    # --- Evaluation prompt ---
    system_prompt = (
        "You are an expert multimodal evaluator. "
        "Given an image, a question, a ground-truth answer, and a model's answer, "
        "judge whether the model's answer is:\n"
        "- 'correct': matches the ground truth factually.\n"
        "- 'grounded': refers only to visible evidence even if not exactly matching.\n"
        "- 'misperception': describes something visible but wrong.\n"
        "- 'hallucination': mentions an object or fact not visible in the image.\n"
        "Respond with only one label word."
    )

    # --- Iterate samples ---
    for item in tqdm(preds, total=len(preds)):
        question = item.get("question")
        model_answer = item.get("model_answer", "")
        gt_answer = item.get("gt_answer", gt_all.get(question, {}).get("gt_answer", ""))
        image_path = item.get("image_path", item.get("image_src", None))

        # --- Encode local image as base64 ---
        image_message = None
        if image_path and os.path.exists(image_path):
            try:
                b64 = encode_image_base64(image_path)
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                }
            except Exception as e:
                print(f"⚠️ Error encoding image {image_path}: {e}")

        # --- Construct GPT-4V message ---
        user_content = [
            {
                "type": "text",
                "text": f"Question: {question}\n"
                        f"Ground Truth: {gt_answer}\n"
                        f"Model Answer: {model_answer}\n"
                        f"Classify the model's answer:"
            }
        ]
        if image_message:
            user_content.append(image_message)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # --- Query GPT-4V ---
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",        # ← full GPT-4V model
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )
            label = resp.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"⚠️ Error evaluating sample: {e}")
            label = "error"

        item["judge"] = label
        results.append(item)

    # --- Compute metrics ---
    stats = {
        "correct": sum(i["judge"] == "correct" for i in results),
        "grounded": sum(i["judge"] == "grounded" for i in results),
        "misperception": sum(i["judge"] == "misperception" for i in results),
        "hallucination": sum(i["judge"] == "hallucination" for i in results),
    }
    total = len(results)
    metrics = {k: round(v / total, 4) for k, v in stats.items()}
    metrics["total"] = total

    print("\n=== GPT-4V Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"{k:15s}: {v}")

    with open(save_path, "w") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)

    print(f"\nSaved GPT-4V judgments to: {save_path}")
    return metrics