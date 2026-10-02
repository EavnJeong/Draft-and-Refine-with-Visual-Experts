from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from .probe_core import (
    LABELS,
    build_panels,
    generate_binding_samples,
    make_composite,
    rule_eval,
    stable_hash,
)
from .qwen_adapter import QwenBinaryScorer


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_binding(
    scorer: QwenBinaryScorer,
    out_dir: Path,
    *,
    limit: int = 16,
    max_permutations: int = 24,
) -> Dict:
    samples = generate_binding_samples(limit)
    perms = list(itertools.permutations(LABELS))[:max_permutations]
    modes = ("text_labels", "in_image_anchor", "composite")
    rows: List[Dict] = []

    for sample in samples:
        for perm in perms:
            physical_pos = {label: i for i, label in enumerate(perm)}
            for mode in modes:
                anchor = mode != "text_labels"
                panels = build_panels(sample.bits, sample.styles, anchor=anchor, subset="ABCD")

                if mode == "composite":
                    composite = make_composite(panels, order=perm)
                    labeled = [("COMPOSITE", composite)]
                else:
                    labeled = [(label, panels[label]) for label in perm]

                for target in LABELS:
                    q = (
                        f"What binary value is encoded by the image/panel labeled {target}? "
                        "Use only the visual cue associated with that label."
                    )
                    r = scorer.score_binary(labeled, q, composite=(mode == "composite"))
                    pred = int(r["prediction"])
                    gt = sample.bits[target]
                    rows.append({
                        "sample_id": sample.sample_id,
                        "mode": mode,
                        "target_label": target,
                        "physical_position": physical_pos[target],
                        "style": sample.styles[target],
                        "ground_truth": gt,
                        "prediction": pred,
                        "correct": int(pred == gt),
                        "logp_0": r["logp_0"],
                        "logp_1": r["logp_1"],
                        "margin": abs(float(r["logp_1"]) - float(r["logp_0"])),
                        "permutation": "".join(perm),
                        "prompt_hash": stable_hash(str(r["prompt"])),
                    })

    write_csv(out_dir / "binding_matrix.csv", rows)

    summaries = {}
    for mode in modes:
        mr = [x for x in rows if x["mode"] == mode]
        by_pos = {}
        by_label = {}
        by_style = {}
        for pos in range(4):
            z = [x["correct"] for x in mr if x["physical_position"] == pos]
            by_pos[str(pos)] = sum(z) / len(z)
        for label in LABELS:
            z = [x["correct"] for x in mr if x["target_label"] == label]
            by_label[label] = sum(z) / len(z)
        for style in sorted(set(x["style"] for x in mr)):
            z = [x["correct"] for x in mr if x["style"] == style]
            by_style[style] = sum(z) / len(z)
        vals = list(by_pos.values())
        summaries[mode] = {
            "mean_accuracy": sum(x["correct"] for x in mr) / len(mr),
            "slot_accuracy": by_pos,
            "label_accuracy": by_label,
            "style_accuracy": by_style,
            "slot_gap": max(vals) - min(vals),
            "min_slot_accuracy": min(vals),
        }

    passing = {
        mode: s for mode, s in summaries.items()
        if s["min_slot_accuracy"] >= 0.90 and s["slot_gap"] <= 0.10
    }
    # Preserve the multi-image scientific setting whenever possible.
    # Composite is a fallback control, not the preferred evidence format.
    if "in_image_anchor" in passing:
        selected_mode = "in_image_anchor"
        scope = "multi_image"
        gate, stop = "PASS", None
    elif "composite" in passing:
        selected_mode = "composite"
        scope = "composite_control_only"
        gate, stop = "PASS", None
    else:
        selected_mode = None
        scope = None
        gate, stop = "FAIL", "BINDING_NOT_NEUTRALIZED"

    summary = {
        "binding_gate": gate,
        "stop_reason": stop,
        "selected_mode": selected_mode,
        "scope": scope,
        "modes": summaries,
        "n_samples": limit,
        "n_permutations": len(perms),
    }
    (out_dir / "binding_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def logic_prompt(rule: str, a: int, b: int) -> str:
    if rule == "xor":
        return f"A={a}, B={b}. Output 1 iff A and B are different; otherwise output 0."
    if rule == "same":
        return f"A={a}, B={b}. Output 1 iff A and B are equal; otherwise output 0."
    if rule == "lookup_xor":
        return (
            f"A={a}, B={b}. Use exactly this lookup table: "
            "00->0, 01->1, 10->1, 11->0. Return the table result."
        )
    raise ValueError(rule)


def run_logic(
    scorer: QwenBinaryScorer,
    out_dir: Path,
    *,
    n: int = 256,
) -> Dict:
    rules = ("xor", "same", "lookup_xor")
    atoms = list(itertools.product((0, 1), repeat=2))
    rows = []
    for rule in rules:
        for i in range(n):
            a, b = atoms[i % len(atoms)]
            gt = rule_eval(rule, a, b)
            q = logic_prompt(rule, a, b)
            r = scorer.score_binary_text(q)
            pred = int(r["prediction"])
            rows.append({
                "rule": rule,
                "sample_id": f"LOGIC_{i:04d}",
                "A": a,
                "B": b,
                "ground_truth": gt,
                "prediction": pred,
                "correct": int(pred == gt),
                "logp_0": r["logp_0"],
                "logp_1": r["logp_1"],
                "margin": abs(float(r["logp_1"]) - float(r["logp_0"])),
            })

    write_csv(out_dir / "logic_calibration.csv", rows)
    stats = {}
    for rule in rules:
        rr = [x for x in rows if x["rule"] == rule]
        stats[rule] = {
            "accuracy": sum(x["correct"] for x in rr) / len(rr),
            "mean_margin": statistics.fmean(x["margin"] for x in rr),
        }

    passing = [(r, s) for r, s in stats.items() if s["accuracy"] >= 0.95]
    if passing:
        passing.sort(key=lambda x: (-x[1]["accuracy"], -x[1]["mean_margin"]))
        selected = passing[0][0]
        gate, stop = "PASS", None
    else:
        selected = None
        gate, stop = "FAIL", "LOGIC_CAPACITY_FAILURE"

    summary = {
        "logic_gate": gate,
        "stop_reason": stop,
        "selected_rule": selected,
        "rules": stats,
        "n": n,
    }
    (out_dir / "logic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["binding", "logic", "both"], default="both")
    ap.add_argument("--output", default="results/e01_probe_v3/calibration")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--logic-n", type=int, default=256)
    ap.add_argument("--max-permutations", type=int, default=24)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scorer = QwenBinaryScorer(args.model_name, args.device, args.cache_dir)

    result = {}
    if args.stage in ("binding", "both"):
        result["binding"] = run_binding(
            scorer, out, limit=args.limit, max_permutations=args.max_permutations
        )
    if args.stage in ("logic", "both"):
        result["logic"] = run_logic(scorer, out, n=args.logic_n)
    (out / "calibration_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
