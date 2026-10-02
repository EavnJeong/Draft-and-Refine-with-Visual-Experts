"""Independent E01 runner using only the repository Qwen loader and scorer."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess

from e01_probe.build_dataset_v2 import LEGENDS, build_samples, write_dataset
from e01_probe.report import summarize
from process.interaction_credit import LABELS, subsets

MODEL_NAME = 'Qwen/Qwen2.5-VL-7B-Instruct'


def question(sample, target, text_only=False):
    if sample['probe'] == 'A':
        task = 'STEP1 = A XOR B. STEP2 = C XOR D. FINAL = STEP1 XOR STEP2.'
    else:
        task = ('A XOR B = Z and C XOR D = Z. STEP = Z. '
                'Either complete pair AB or CD determines Z.')
    task += '\nXOR is 1 when its two inputs differ and 0 when they are equal.'
    if text_only:
        evidence = ', '.join(f'{i}={sample["bits"][i]}' for i in LABELS)
    else:
        evidence = '\n'.join(f'Image {i} uses {sample["cue_styles"][i]}: '
                             f'{LEGENDS[sample["cue_styles"][i]]}.' for i in LABELS)
        evidence += '\nMissing images are unknown bits; do not interpret absence as zero.'
    return f'{task}\n{evidence}\nWhat is {target}?'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen_vl'], default='qwen_vl')
    parser.add_argument('--model-name', default=MODEL_NAME)
    parser.add_argument('--probe', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cache-dir')
    parser.add_argument('--output-dir', type=Path, default=Path('results/e01_probe_v2'))
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    parser.add_argument('--ability-threshold', type=float, default=.8)
    parser.add_argument('--loo-margin', type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.ability_threshold <= 1 or args.loo_margin <= 0 or args.bootstrap_draws < 100:
        parser.error('Require 0 < ability-threshold <= 1, loo-margin > 0, bootstrap-draws >= 100')
    probes = ('A', 'B') if args.probe == 'both' else (args.probe,)
    # Build and validate the full design, then take complete truth-table blocks.
    samples = []
    for probe in probes:
        count = max(128, args.limit)
        full = build_samples(probe, count, args.seed)
        build_samples(probe, args.limit, args.seed)  # validates smoke sample size
        samples.extend(full[:args.limit])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_dataset(samples, args.output_dir / 'dataset')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from models.vlm.loader import load_qwen25_vl
    from models.vlm.qwen_vl_predict import score_binary_candidates_qwen

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, processor = load_qwen25_vl(model_name=args.model_name, device=args.device,
                                     cache_dir=args.cache_dir, attn_implementation='eager')
    model.eval()
    manifest = args.output_dir / 'dataset' / 'manifest.json'
    metadata = {**vars(args), 'output_dir': str(args.output_dir),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'model_revision': getattr(model.config, '_commit_hash', None),
                'dataset_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'Pillow')},
                'status': 'RUNNING', 'candidate_normalization': 'log-softmax over scores for 0 and 1',
                'attention_implementation': 'eager', 'dtype': str(model.dtype)}
    metadata_path = args.output_dir / 'run.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    fields = ['sample_id', 'probe', 'target_field', 'subset', 'ground_truth', 'logp_0', 'logp_1',
              'p_0', 'p_1', 'candidate_loglik_0', 'candidate_loglik_1', 'p_correct',
              'prediction', 'correct', 'cue_styles', 'model_name', 'seed', 'empty_mode', 'scoring_mode']
    raw, controls = [], []
    with (args.output_dir / 'raw_scores.csv').open('w', newline='') as raw_file, \
            (args.output_dir / 'controls.csv').open('w', newline='') as control_file:
        raw_writer = csv.DictWriter(raw_file, fieldnames=fields)
        control_writer = csv.DictWriter(control_file, fieldnames=fields + ['control'])
        raw_writer.writeheader()
        control_writer.writeheader()

        def score(sample, target, truth, subset, prompt, images, control=None):
            result = score_binary_candidates_qwen(model, processor, prompt, images, subset)
            prediction = int(result['logp_1'] > result['logp_0'])
            row = {**result, 'sample_id': sample['sample_id'], 'probe': sample['probe'],
                   'target_field': target, 'subset': subset, 'ground_truth': truth,
                   'p_correct': result[f'p_{truth}'], 'prediction': prediction,
                   'correct': int(prediction == truth), 'cue_styles': json.dumps(sample['cue_styles'], sort_keys=True),
                   'model_name': args.model_name, 'seed': args.seed}
            if control:
                row['control'] = control
                controls.append(row)
                control_writer.writerow(row)
                control_file.flush()
            else:
                raw.append(row)
                raw_writer.writerow(row)
                raw_file.flush()

        for index, sample in enumerate(samples):
            images = {i: args.output_dir / 'dataset' / f"{sample['sample_id']}_{i}.png" for i in LABELS}
            for target, truth in sample['targets'].items():
                score(sample, target, truth, 'EMPTY', question(sample, target, True), {}, 'text_only_logic')
                for subset in subsets():
                    score(sample, target, truth, subset, question(sample, target), images)
            for label in LABELS:
                prompt = (f'Image {label}: {LEGENDS[sample["cue_styles"][label]]}. '
                          f'What binary value does Image {label} encode?')
                score(sample, label, sample['bits'][label], label, prompt, images, 'single_cue')
            print(f"[{index + 1}/{len(samples)}] {sample['sample_id']}", flush=True)
    summary, credits = summarize(raw, controls, args.seed, args.bootstrap_draws,
                                 args.ability_threshold, args.loo_margin)
    with (args.output_dir / 'interactions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(credits[0]))
        writer.writeheader()
        writer.writerows(credits)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    metadata['status'] = 'COMPLETE'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'gates': summary['gates'], 'STOP_REASON': summary['STOP_REASON'],
                      'overall_status': summary['overall_status']}, indent=2))


if __name__ == '__main__':
    main()
