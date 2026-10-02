# E01 Probe-v3: binding-neutralized causal probe

This branch isolates three prerequisites before making any claim about coalition credit:

1. **Binding neutralization** — measure physical-slot/image-label bias under permutations.
2. **Logic calibration** — select a one-step binary rule that the text-only model solves at >=95%.
3. **Fixed-slot causal lattice** — all 16 evidence subsets preserve the same visual slots and prompt structure; missing evidence is replaced by a neutral blank panel.

No training, LoRA, RL, SAM, GroundingDINO, CLIPSeg, or DnR expert is used.

## Why Probe-v2 was inconclusive

Probe-v2 failed the logic prerequisite and exposed a strong physical-position/image-label effect. If removing an evidence image also shifts the remaining images to new physical positions, the measured set function is not invariant:

```
v_tilde(S) = E_{phi ~ q_S}[v(S; phi)]
```

with subset-dependent binding state `q_S`. Harsanyi/LOO values from different subsets then mix evidence removal with position/binding changes.

Probe-v3 fixes this by keeping slot count, order, label anchors, and prompt template constant.

## Environment

Use the existing project environment. The code reuses:

```
models/vlm/loader.py::load_qwen25_vl
```

and does not change upstream DnR files.

## CPU analytical tests

```bash
pytest -q tests/test_e01_probe_v3.py
```

The tests verify the redundant-coalition oracle:

```
LOO_A = LOO_B = LOO_C = LOO_D = 0
Gamma_AB = Gamma_CD = ln(2)
wrong-pair Gamma = 0
```

## Stage 1: binding + logic smoke test

```bash
python -m e01_probe_v3.binding_logic \
  --stage both \
  --output results/e01_probe_v3/calibration_smoke \
  --limit 16 \
  --logic-n 32 \
  --max-permutations 24 \
  --device cuda:0
```

Binding formats:

- `text_labels`: textual image-slot labels only.
- `in_image_anchor`: A/B/C/D is drawn inside each image while retaining four images.
- `composite`: fixed 2x2 labeled canvas; this is a fallback control only.

Gate:

- every physical slot accuracy >= 0.90;
- slot gap <= 0.10;
- one-step text-only rule accuracy >= 0.95.

`in_image_anchor` is preferred whenever it passes. Composite passing alone is recorded as `composite_control_only`.

## Stage 2: full calibration

```bash
python -m e01_probe_v3.binding_logic \
  --stage both \
  --output results/e01_probe_v3/calibration \
  --limit 64 \
  --logic-n 256 \
  --max-permutations 24 \
  --device cuda:0
```

## Stage 3: fixed-slot coalition smoke test

Only run after both calibration gates pass.

```bash
python -m e01_probe_v3.coalition_probe \
  --calibration-dir results/e01_probe_v3/calibration \
  --output results/e01_probe_v3/coalition_smoke \
  --n 16 \
  --binding-shift-limit 8 \
  --device cuda:0
```

## Stage 4: full coalition probe

```bash
python -m e01_probe_v3.coalition_probe \
  --calibration-dir results/e01_probe_v3/calibration \
  --output results/e01_probe_v3/coalition \
  --n 128 \
  --binding-shift-limit 16 \
  --device cuda:0
```

The coalition runner additionally measures subset-induced binding shift: for a surviving cue, does its decoded value change merely because another slot was blanked? If the rate exceeds 0.05, interaction results are marked contaminated.

## Interpretation

A PASS is still only a phenomenon gate. Continue E01 only if:

- binding and text-only logic prerequisites pass;
- fixed-slot lattice is complete;
- subset-induced binding shift <= 0.05;
- paired bootstrap CI for correct-pair Gamma minus wrong-pair Gamma is strictly above zero.

Do not implement a new loss or RL objective before those conditions hold.
