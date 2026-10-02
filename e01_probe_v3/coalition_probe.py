from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .binding_logic import logic_prompt
from .probe_core import (
    LABELS,
    SUBSET_NAMES,
    bootstrap_ci,
    build_panels,
    correct_pairs,
    generate_coalition_samples,
    harsanyi_dividend,
    leave_one_out,
    make_composite,
    stable_hash,
    wrong_pairs,
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


def coalition_question(rule: str) -> str:
    if rule in ("xor", "lookup_xor"):
        core = (
            "A and B encode one redundant route to the target bit, and C and D encode "
            "a second redundant route. The target equals XOR(A,B) and also XOR(C,D)."
        )
    elif rule == "same":
        core = (
            "A and B encode one redundant route to the target bit, and C and D encode "
            "a second redundant route. The target is 1 iff A and B are equal, and is "
            "the same result for C and D."
        )
    else:
        raise ValueError(rule)
    return core + " Return the target bit."


def build_input(sample, subset: str, mode: str):
    anchor = mode in ("in_image_anchor", "composite")
    panels = build_panels(sample.bits, sample.styles, anchor=anchor, subset=subset)
    if mode == "composite":
        return [("COMPOSITE", make_composite(panels))], True
    return [(label, panels[label]) for label in LABELS], False


def run_coalition(
    scorer: QwenBinaryScorer,
    out_dir: Path,
    *,
    mode: str,
    rule: str,
    n: int = 128,
    binding_shift_limit: int = 16,
) -> Dict:
    samples = generate_coalition_samples(n, rule)
    raw: List[Dict] = []

    for sample in samples:
        for subset in SUBSET_NAMES:
            labeled, composite = build_input(sample, subset, mode)
            q = coalition_question(rule)
            r = scorer.score_binary(labeled, q, composite=composite)
            gt = sample.target
            logp_correct = float(r[f"logp_{gt}"])
            raw.append({
                "sample_id": sample.sample_id,
                "subset": subset,
                "ground_truth": gt,
                "prediction": int(r["prediction"]),
                "correct": int(int(r["prediction"]) == gt),
                "logp_0": r["logp_0"],
                "logp_1": r["logp_1"],
                "logp_correct": logp_correct,
                "model_name": scorer.model_name,
                "mode": mode,
                "rule": rule,
                "image_count": 1 if composite else 4,
                "semantic_labels": "ABCD",
                "physical_slots": "ABCD" if not composite else "2x2_ABCD",
                "intervention_mode": "fixed_slot",
                "prompt_hash": stable_hash(str(r["prompt"])),
            })

    write_csv(out_dir / "raw_scores.csv", raw)

    grouped = defaultdict(dict)
    for row in raw:
        grouped[row["sample_id"]][row["subset"]] = float(row["logp_correct"])

    interactions: List[Dict] = []
    for sample in samples:
        vals = grouped[sample.sample_id]
        if set(vals) != set(SUBSET_NAMES):
            raise RuntimeError(f"Incomplete lattice for {sample.sample_id}")
        rec = {
            "sample_id": sample.sample_id,
            "ground_truth": sample.target,
        }
        for item in LABELS:
            rec[f"LOO_{item}"] = leave_one_out(vals, item)
        for a, b in correct_pairs() + wrong_pairs():
            rec[f"Gamma_{a}{b}"] = harsanyi_dividend(vals, (a, b))
        rec["Gamma_ABCD"] = harsanyi_dividend(vals, LABELS)
        interactions.append(rec)

    write_csv(out_dir / "interactions.csv", interactions)

    # Subset-induced binding shift: on a small prespecified sample, blank other slots
    # and verify that a surviving cue's decoded value does not change.
    shift_rows = []
    for sample in samples[:binding_shift_limit]:
        full_input, full_comp = build_input(sample, "ABCD", mode)
        full_pred = {}
        for label in LABELS:
            q = f"What binary value is encoded by the image/panel labeled {label}?"
            r = scorer.score_binary(full_input, q, composite=full_comp)
            full_pred[label] = int(r["prediction"])

        for subset in SUBSET_NAMES:
            keep = set() if subset == "EMPTY" else set(subset)
            if not keep:
                continue
            labeled, comp = build_input(sample, subset, mode)
            for label in sorted(keep):
                q = f"What binary value is encoded by the image/panel labeled {label}?"
                r = scorer.score_binary(labeled, q, composite=comp)
                pred = int(r["prediction"])
                shift_rows.append({
                    "sample_id": sample.sample_id,
                    "subset": subset,
                    "label": label,
                    "full_prediction": full_pred[label],
                    "subset_prediction": pred,
                    "changed": int(pred != full_pred[label]),
                    "ground_truth": sample.bits[label],
                })

    write_csv(out_dir / "binding_shift.csv", shift_rows)
    binding_shift = (
        sum(x["changed"] for x in shift_rows) / len(shift_rows)
        if shift_rows else float("nan")
    )

    def mean(key):
        return sum(float(x[key]) for x in interactions) / len(interactions)

    correct_gamma = [mean("Gamma_AB"), mean("Gamma_CD")]
    wrong_gamma = [mean(f"Gamma_{a}{b}") for a, b in wrong_pairs()]
    loo_abs = sum(abs(mean(f"LOO_{x}")) for x in LABELS) / 4

    # Per-sample paired contrast: average correct pair gamma - average wrong pair gamma.
    contrasts = []
    for x in interactions:
        c = (x["Gamma_AB"] + x["Gamma_CD"]) / 2
        w = sum(x[f"Gamma_{a}{b}"] for a, b in wrong_pairs()) / 4
        contrasts.append(c - w)
    ci_lo, ci_hi = bootstrap_ci(contrasts)

    image_count_ok = all(
        x["image_count"] == (1 if mode == "composite" else 4)
        for x in raw
    )
    causal_gate = image_count_ok and binding_shift <= 0.05
    coalition_gate = causal_gate and ci_lo > 0.0

    summary = {
        "mode": mode,
        "rule": rule,
        "n": n,
        "causal_lattice_gate": "PASS" if causal_gate else "FAIL",
        "coalition_signal_gate": "PASS" if coalition_gate else "FAIL",
        "binding_shift_rate": binding_shift,
        "mean_abs_loo": loo_abs,
        "mean_gamma_AB": correct_gamma[0],
        "mean_gamma_CD": correct_gamma[1],
        "mean_wrong_pair_gamma": sum(wrong_gamma) / len(wrong_gamma),
        "correct_minus_wrong_gamma_mean": sum(contrasts) / len(contrasts),
        "correct_minus_wrong_gamma_ci95": [ci_lo, ci_hi],
        "stop_reason": (
            None if coalition_gate else
            "BINDING_CONTAMINATED" if not causal_gate else
            "NO_COALITION_SIGNAL"
        ),
    }
    (out_dir / "coalition_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-dir", default="results/e01_probe_v3/calibration")
    ap.add_argument("--output", default="results/e01_probe_v3/coalition")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--binding-shift-limit", type=int, default=16)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cal = Path(args.calibration_dir)
    binding = json.loads((cal / "binding_summary.json").read_text())
    logic = json.loads((cal / "logic_summary.json").read_text())
    if binding["binding_gate"] != "PASS":
        raise SystemExit("STOP: BINDING_NOT_NEUTRALIZED")
    if logic["logic_gate"] != "PASS":
        raise SystemExit("STOP: LOGIC_CAPACITY_FAILURE")

    scorer = QwenBinaryScorer(args.model_name, args.device, args.cache_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = run_coalition(
        scorer,
        out,
        mode=binding["selected_mode"],
        rule=logic["selected_rule"],
        n=args.n,
        binding_shift_limit=args.binding_shift_limit,
    )
    gate = {
        "BINDING_GATE": binding["binding_gate"],
        "LOGIC_GATE": logic["logic_gate"],
        "CAUSAL_LATTICE_GATE": summary["causal_lattice_gate"],
        "COALITION_SIGNAL_GATE": summary["coalition_signal_gate"],
        "overall_status": (
            "PASS" if summary["coalition_signal_gate"] == "PASS" else "FAIL"
        ),
        "stop_reason": summary["stop_reason"],
    }
    (out / "gate_report.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
