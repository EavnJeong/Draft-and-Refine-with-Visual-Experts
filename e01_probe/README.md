# E01: step-conditioned coalition credit falsification probe

This is an isolated inference experiment. It reuses `load_qwen25_vl()` and adds
candidate scoring beside the existing inference functions. The DnR entry points,
expert loaders, training, and model weights are unchanged.

## Server environment

Use Python 3.10+ with a CUDA-enabled PyTorch 2.8.0 / torchvision 0.23.0 environment.
Install the small probe dependency list (the root requirements file is a package
inventory, not a pip requirements file):

```bash
python -m pip install -r e01_probe/requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

Default model: `Qwen/Qwen2.5-VL-7B-Instruct`, BF16, one GPU, eager attention.
Use `--model-name /absolute/path/to/Qwen2.5-VL-7B-Instruct` for offline weights,
or `--cache-dir /path/to/hf-cache`. No visual expert dependency is needed.

## Smoke tests (32 samples per probe)

```bash
python probe_e01_v2.py --model qwen_vl --probe A --limit 32 --device cuda:0 --seed 42 --output-dir results/e01_probe_v2/smoke_A
python probe_e01_v2.py --model qwen_vl --probe B --limit 32 --device cuda:0 --seed 42 --output-dir results/e01_probe_v2/smoke_B
```

A writes 1,536 interventions and 224 control scores. B writes 512 interventions
and 160 control scores. Each output directory must be new: reruns cannot silently
overwrite earlier evidence. A-only reports G4 as NOT_RUN; B-only reports G2/G3 as
NOT_RUN. Inspect the applicable gates and STOP_REASON in each summary before
proceeding. An exit code of zero means the run completed, not that gates passed.

For the full 128-sample experiment, run both probes together for all four gates:

```bash
python probe_e01_v2.py --model qwen_vl --probe both --limit 128 --device cuda:0 --seed 42 --output-dir results/e01_probe_v2/full
```

Without `--output-dir`, the requested output is
`results/e01_probe_v2/raw_scores.csv` (and adjacent files). Choose a different,
new directory if that default directory already exists.

## Design and outputs

Each run writes:

- `dataset/manifest.json` and four 224x224 PNGs per sample: original bit values,
  targets, cue assignments, and images for reproduction.
- `raw_scores.csv`: all 16 subsets per sample and independent target question.
  `logp_0/logp_1` are log-softmax normalized over the two candidate sequence
  likelihoods; `candidate_loglik_0/1` retain the original vocabulary-normalized
  sequence log likelihoods. `p_correct`, predictions, styles, seed, model name,
  `empty_mode`, and `scoring_mode` are recorded for every call.
- `controls.csv`: raw text-only logic and single-image perception scores.
- `interactions.csv`: all 15 nonempty Harsanyi dividends and four LOO values per
  sample/target. First-order dividends equal standalone marginals.
- `summary.json`: means, pointwise bootstrap CIs, paired contrasts, gate decisions,
  thresholds, and stop reasons. Missing probes have null/empty metrics and NOT_RUN.
- `run.json`: arguments, dependency versions, model revision (when available),
  Git commit, dataset manifest SHA256, dtype, and completion status. Interrupted
  scoring retains flushed raw rows but has no final summary.

A uses complete 16-row truth-table blocks; B uses complete 8-row blocks. A local
seeded RNG shuffles each block. Cue styles rotate across blocks, independently of
all bit tuples. Balance is asserted globally and within every style assignment.
At least 128 samples are constructed/validated; smoke runs take a balanced prefix.
All four styles appear in every sample. With A's 32-sample smoke, each label sees
two styles; 128 samples cover the full four-style rotation for each label.

Visual prompts contain rules and cue legends, never the actual bit values. Image
labels precede their image tokens in canonical order. Missing bits are unknown.
Text-only controls give all four bits explicitly; perception controls give just
one image and its legend. EMPTY first uses a genuinely image-free processor call.
Only a processor error explicitly indicating missing required images triggers a
fixed gray image fallback, recorded as `empty_mode=blank_image`.

Scoring checks tokenizations of `0` and `1`. Single-token candidates use the last
prompt-position logits. Otherwise the exact candidate IDs are appended to the
processed prompt and only their conditional token log likelihoods are summed.
No terminal EOS is included in the candidate. The model is in evaluation mode;
RNGs are seeded, deterministic PyTorch algorithms enabled, and TF32 disabled.
Hardware/library changes can still alter numerical results; unsupported
deterministic operations fail rather than silently run nondeterministically.

## Fixed gate rules

Effects use `v(S) = log p(correct | S)` in **nats**. Defaults are ability accuracy
0.8 and equivalence margin 0.05 nats; override only before inspecting outcomes,
using `--ability-threshold` and `--loo-margin`. Bootstrap uses 2,000 seeded
resamples of samples, retaining each within-sample contrast; all CIs are
pointwise percentile 95%, not simultaneous confidence guarantees. Repeated
synthetic designs do not establish generalization to natural images, or variation
across model/training seeds.

- **G1_BASIC_ABILITY:** full-evidence accuracy for every target, text logic for
  every target, and perception for every label must have CI lower bounds >= 0.8.
  A CI upper bound < 0.8 fails; intermediate cases are INCONCLUSIVE.
- **G2_STEP_IDENTIFIABILITY:** A's AB dividend for STEP1 and CD dividend for STEP2
  must each have a positive CI lower bound, as must every paired difference from
  each of the five wrong pairs.
- **G3_FINAL_ONLY_SEPARATION:** A's FINAL four-way dividend must be positive and
  significantly exceed every first-, second-, and third-order dividend by paired
  contrasts. This tests separation at FINAL versus the step-specific results in G2.
- **G4_LOO_SEPARATION:** B's four LOO CIs must be entirely inside +/-0.05 nats;
  AB/CD must each be positive and exceed each wrong pair by paired contrasts.
  Wrong-pair CIs must also be entirely inside the same equivalence interval.

A positive-effect CI with upper bound <= 0 fails; a CI spanning zero is
INCONCLUSIVE. LOO fails only if its CI lies beyond the equivalence tolerance;
a wide CI containing zero is INCONCLUSIVE, never proof of equivalence.
STOP_REASON prioritizes LOGIC_FAILURE, PERCEPTION_FAILURE, NO_STEP_INTERACTION,
PROBE_LEAKAGE, then NO_COALITION_SIGNAL. Other unresolved gates remain visible in
`gates` and `overall_status`. No failure is attributed to visual integration when
basic logic or perception is insufficient. All requested scores are collected
before reporting gates; STOP_REASON is a recommendation to stop further stages.

The CPU tests enumerate an analytical Bayes scorer over each finite population.
They validate the ln(2) dividends, zero wrong pairs, zero B LOO, dataset balance,
Möbius reconstruction, scorer alignment/fallback, and reporting. They do not
establish real Qwen accuracy or GPU compatibility; that requires the server runs.
