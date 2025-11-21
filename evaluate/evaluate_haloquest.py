import json
import numpy as np
from collections import defaultdict
from openai import OpenAI
import time
import os
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def build_batch_prompt(batch):
    """
    Build a single batch prompt for multiple questions.
    """
    task_intro = """You are a visual reasoning evaluator.

For each example, decide whether the model's answer is a 'hallucination', 'misperception', or 'not'.

Definitions:
- "hallucination": mentions an object, attribute, or action that does NOT exist or is unsupported by the image.
- "misperception": refers to a real visual concept but with wrong details (e.g., wrong color, count, or attribute).
- "not": consistent with the groundtruth description.

Respond with exactly one line per example in the format:
Example i: <hallucination/misperception/not>
"""

    parts = []
    for i, (htype, question, pred, gt) in enumerate(batch):
        parts.append(f"\nExample {i+1}:\nQuestion: {question}\nModel answer: {pred}\nGroundtruth answers: {gt}")
    return task_intro + "\n".join(parts)


def judge_hallucination_batch(batch):
    """
    Batch version of hallucination judgment.
    - Takes a list of (htype, question, pred, gt)
    - Returns a list of results of same length.
    """
    prompt = build_batch_prompt(batch)

    while True:
        for model_name in ["gpt-5-mini", "gpt-4o-mini"]:
            try:
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = r.choices[0].message.content.lower().strip()
                lines = [l.strip() for l in text.splitlines() if l.strip()]

                results = []
                for l in lines:
                    if "halluc" in l:
                        results.append("hallucination")
                    elif "misp" in l:
                        results.append("misperception")
                    elif "not" in l:
                        results.append("not")
                # ensure alignment
                while len(results) < len(batch):
                    results.append("not")
                return results[:len(batch)]

            except Exception as e:
                msg = str(e).lower()
                if "rate_limit" in msg or "429" in msg:
                    print(f"⚠️ [{model_name}] Rate limit hit, switching model or retrying...")
                    continue
                elif "timeout" in msg or "overloaded" in msg:
                    print(f"⚠️ [{model_name}] Temporary server issue, retrying...")
                    continue
                else:
                    print(f"⚠️ [{model_name}] Unexpected error: {e}")
                    continue
        print("⏳ All models failed. Waiting 10 seconds before retrying...")
        time.sleep(10)


def eval_haloquest(result_path, annfile=None, batch_size=10):
    """
    Evaluate hallucination/misperception/correct ratios per type (batched).
    """
    with open(result_path, "r") as f:
        data = json.load(f)

    type_groups = defaultdict(list)
    for x in data:
        htype = x.get("hallucination_type", "unknown").lower().strip()
        type_groups[htype].append(x)

    print(f"[HaloQuest Evaluation] Loaded {len(data)} samples.")
    print("--------------------------------------------------")

    overall_stats = {}

    for htype, items in type_groups.items():
        print(f"\n=== [TYPE: {htype.upper()}] ===")
        results = []

        for i in range(0, len(items), batch_size):
            batch_items = items[i:i+batch_size]
            batch_input = [(htype, it["question"], it["pred"], it["answers"]) for it in batch_items]
            batch_results = judge_hallucination_batch(batch_input)
            results.extend(batch_results)

        hallu_count = sum(r == "hallucination" for r in results)
        mis_count = sum(r == "misperception" for r in results)
        not_count = sum(r == "not" for r in results)
        total = len(results)

        print("--------------------------------------------------")
        print(f"Type: {htype}")
        print(f"  • Total samples     : {total}")
        print(f"  • Hallucinations    : {hallu_count} ({hallu_count/total*100:.2f}%)")
        print(f"  • Misperceptions    : {mis_count} ({mis_count/total*100:.2f}%)")
        print(f"  • Correct (not)     : {not_count} ({not_count/total*100:.2f}%)")
        print("--------------------------------------------------")

        overall_stats[htype] = {
            "hallucination_rate": hallu_count / total,
            "misperception_rate": mis_count / total,
            "correct_rate": not_count / total,
            "n": total
        }

    # === Overall average ===
    mean_h = np.mean([v["hallucination_rate"] for v in overall_stats.values()])
    mean_m = np.mean([v["misperception_rate"] for v in overall_stats.values()])
    mean_c = np.mean([v["correct_rate"] for v in overall_stats.values()])

    print("\n==================================================")
    print(f"[OVERALL AVERAGE RATES]")
    print(f"  • Hallucination : {mean_h*100:.2f}%")
    print(f"  • Misperception : {mean_m*100:.2f}%")
    print(f"  • Correct       : {mean_c*100:.2f}%")
    print("==================================================")

    return overall_stats