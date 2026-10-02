"""Balanced finite populations; cue assignment is independent of every bit tuple."""
import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw

from process.interaction_credit import LABELS, subsets

STYLES = ('position', 'color', 'size', 'orientation')
LEGENDS = {
    'position': 'black circle on the left = 0; on the right = 1',
    'color': 'red circle = 0; blue circle = 1',
    'size': 'small black circle = 0; large black circle = 1',
    'orientation': 'horizontal black bar = 0; vertical black bar = 1',
}


def entropy(values):
    counts = Counter(values)
    return -sum((n / len(values)) * math.log2(n / len(values)) for n in counts.values())


def conditional_entropy(rows, target, labels):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row['bits'][i] for i in labels)].append(row['targets'][target])
    return sum(len(v) / len(rows) * entropy(v) for v in groups.values())


def mutual_information(rows, target, labels):
    return entropy([r['targets'][target] for r in rows]) - conditional_entropy(rows, target, labels)


def validate(rows, probe):
    """Also validate within each cue assignment, excluding style-based leakage."""
    targets = ('STEP1', 'STEP2', 'FINAL') if probe == 'A' else ('STEP',)
    for target in targets:
        assert abs(entropy([r['targets'][target] for r in rows]) - 1) < 1e-12
    if probe == 'A':
        for target, pair in [('STEP1', 'AB'), ('STEP2', 'CD')]:
            for i in LABELS:
                assert abs(mutual_information(rows, target, i)) < 1e-12
            assert abs(conditional_entropy(rows, target, pair)) < 1e-12
        for s in subsets():
            if s != 'ABCD':
                assert abs(conditional_entropy(rows, 'FINAL', '' if s == 'EMPTY' else s) - 1) < 1e-12
    else:
        for i in LABELS:
            assert abs(mutual_information(rows, 'STEP', i)) < 1e-12
        for pair in ['AB', 'CD']:
            assert abs(mutual_information(rows, 'STEP', pair) - 1) < 1e-12
        for pair in ['AC', 'AD', 'BC', 'BD']:
            assert abs(mutual_information(rows, 'STEP', pair)) < 1e-12


def build_samples(probe, count=128, seed=42):
    if probe not in ('A', 'B'):
        raise ValueError('probe must be A or B')
    block_size = 16 if probe == 'A' else 8
    if count < block_size or count % block_size:
        raise ValueError(f'count must be a positive multiple of {block_size}')
    rng = random.Random(seed)
    rows = []
    for block in range(count // block_size):
        tuples = list(product((0, 1), repeat=4 if probe == 'A' else 3))
        rng.shuffle(tuples)
        styles = {label: STYLES[(i + block) % 4] for i, label in enumerate(LABELS)}
        for bits in tuples:
            if probe == 'A':
                a, b, c, d = bits
                targets = {'STEP1': a ^ b, 'STEP2': c ^ d, 'FINAL': a ^ b ^ c ^ d}
            else:
                z, a, c = bits
                b, d = a ^ z, c ^ z
                targets = {'STEP': z}
            rows.append({'sample_id': f'{probe}_{len(rows):04d}', 'probe': probe,
                         'bits': dict(zip(LABELS, (a, b, c, d))),
                         'targets': targets, 'cue_styles': styles.copy()})
    validate(rows, probe)
    for assignment in {tuple(r['cue_styles'].values()) for r in rows}:
        validate([r for r in rows if tuple(r['cue_styles'].values()) == assignment], probe)
    return rows


def render_cue(bit, style):
    image = Image.new('RGB', (224, 224), 'white')
    draw = ImageDraw.Draw(image)
    if style == 'position':
        x = 56 if bit == 0 else 168
        draw.ellipse((x - 24, 88, x + 24, 136), fill='black')
    elif style == 'color':
        draw.ellipse((64, 64, 160, 160), fill='red' if bit == 0 else 'blue')
    elif style == 'size':
        r = 18 if bit == 0 else 70
        draw.ellipse((112 - r, 112 - r, 112 + r, 112 + r), fill='black')
    elif style == 'orientation':
        draw.rectangle((42, 100, 182, 124) if bit == 0 else (100, 42, 124, 182), fill='black')
    else:
        raise ValueError(style)
    return image


def write_dataset(rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for row in rows:
        for label in LABELS:
            render_cue(row['bits'][label], row['cue_styles'][label]).save(
                directory / f"{row['sample_id']}_{label}.png")
    (directory / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', choices=['A', 'B'], required=True)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_dataset(build_samples(args.probe, args.count, args.seed), args.output)
