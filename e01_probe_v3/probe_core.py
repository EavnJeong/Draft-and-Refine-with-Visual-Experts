from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

LABELS = ("A", "B", "C", "D")
STYLES = ("position", "color", "size", "orientation")
ALL_SUBSETS = tuple(
    "".join(c)
    for r in range(5)
    for c in itertools.combinations(LABELS, r)
)
SUBSET_NAMES = tuple("EMPTY" if s == "" else s for s in ALL_SUBSETS)


def subset_to_set(name: str) -> set[str]:
    if name in ("", "EMPTY"):
        return set()
    return set(name)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def render_cue(
    bit: int,
    style: str,
    label: str | None = None,
    *,
    anchor: bool = False,
    size: int = 128,
) -> Image.Image:
    if bit not in (0, 1):
        raise ValueError(bit)
    if style not in STYLES:
        raise ValueError(style)

    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)

    if style == "position":
        x = int(size * (0.28 if bit == 0 else 0.72))
        y = int(size * 0.54)
        r = int(size * 0.11)
        d.ellipse((x-r, y-r, x+r, y+r), fill="black", outline="black")
    elif style == "color":
        fill = (40, 100, 220) if bit == 0 else (230, 130, 30)
        d.polygon(
            [(size//2, int(size*.22)), (int(size*.25), int(size*.76)), (int(size*.75), int(size*.76))],
            fill=fill,
            outline="black",
        )
    elif style == "size":
        s = int(size * (0.18 if bit == 0 else 0.40))
        cx = cy = size // 2
        d.rectangle((cx-s//2, cy-s//2, cx+s//2, cy+s//2), fill=(70, 160, 90), outline="black")
    elif style == "orientation":
        if bit == 0:
            pts = [(size//2, int(size*.18)), (int(size*.27), int(size*.76)), (int(size*.73), int(size*.76))]
        else:
            pts = [(int(size*.27), int(size*.29)), (int(size*.73), int(size*.29)), (size//2, int(size*.82))]
        d.polygon(pts, fill=(150, 90, 190), outline="black")

    if anchor and label:
        # Constant, high-visibility label anchor: metadata, not task evidence.
        font = ImageFont.load_default(size=max(18, size // 7))
        d.rectangle((2, 2, 38, 34), fill="white", outline="black", width=2)
        d.text((10, 5), label, fill="black", font=font)
    return im


def render_blank(label: str, *, anchor: bool, size: int = 128) -> Image.Image:
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    # Keep a faint neutral frame so blank slots have stable visual extent.
    d.rectangle((18, 28, size-18, size-18), outline=(220, 220, 220), width=2)
    if anchor:
        font = ImageFont.load_default(size=max(18, size // 7))
        d.rectangle((2, 2, 38, 34), fill="white", outline="black", width=2)
        d.text((10, 5), label, fill="black", font=font)
    return im


def make_composite(
    panels: Mapping[str, Image.Image],
    order: Sequence[str] = LABELS,
    *,
    gap: int = 8,
    background: str = "white",
) -> Image.Image:
    first = panels[order[0]]
    w, h = first.size
    canvas = Image.new("RGB", (2*w + 3*gap, 2*h + 3*gap), background)
    coords = [
        (gap, gap),
        (2*gap+w, gap),
        (gap, 2*gap+h),
        (2*gap+w, 2*gap+h),
    ]
    for label, xy in zip(order, coords):
        canvas.paste(panels[label], xy)
    return canvas


def rotated_style_map(index: int) -> Dict[str, str]:
    return {
        label: STYLES[(j + index) % len(STYLES)]
        for j, label in enumerate(LABELS)
    }


@dataclass(frozen=True)
class BindingSample:
    sample_id: str
    bits: Dict[str, int]
    styles: Dict[str, str]


@dataclass(frozen=True)
class CoalitionSample:
    sample_id: str
    target: int
    bits: Dict[str, int]
    styles: Dict[str, str]


def generate_binding_samples(n: int, seed: int = 3407) -> List[BindingSample]:
    # Cycle over all 16 bit patterns for exact balance; seed only changes ordering.
    base = list(itertools.product((0, 1), repeat=4))
    rng = random.Random(seed)
    rng.shuffle(base)
    rows: List[BindingSample] = []
    for i in range(n):
        bits_tuple = base[i % len(base)]
        bits = dict(zip(LABELS, bits_tuple))
        rows.append(BindingSample(f"BIND_{i:04d}", bits, rotated_style_map(i)))
    return rows


def rule_eval(rule: str, a: int, b: int) -> int:
    if rule in ("xor", "lookup_xor"):
        return int(a != b)
    if rule == "same":
        return int(a == b)
    raise ValueError(rule)


def generate_coalition_samples(
    n: int,
    rule: str,
    seed: int = 9917,
) -> List[CoalitionSample]:
    # Each pair independently encodes the same target under the selected one-step rule.
    rng = random.Random(seed)
    atoms = list(itertools.product((0, 1), repeat=3))  # target, A, C
    rng.shuffle(atoms)
    rows: List[CoalitionSample] = []
    for i in range(n):
        target, a, c = atoms[i % len(atoms)]
        if rule in ("xor", "lookup_xor"):
            b = a ^ target
            d = c ^ target
        elif rule == "same":
            b = a if target == 1 else 1 - a
            d = c if target == 1 else 1 - c
        else:
            raise ValueError(rule)
        bits = {"A": a, "B": b, "C": c, "D": d}
        assert rule_eval(rule, a, b) == target
        assert rule_eval(rule, c, d) == target
        rows.append(CoalitionSample(f"COAL_{i:04d}", target, bits, rotated_style_map(i)))
    return rows


def build_panels(
    bits: Mapping[str, int],
    styles: Mapping[str, str],
    *,
    anchor: bool,
    subset: str = "ABCD",
    size: int = 128,
) -> Dict[str, Image.Image]:
    keep = subset_to_set(subset)
    result = {}
    for label in LABELS:
        if label in keep:
            result[label] = render_cue(bits[label], styles[label], label, anchor=anchor, size=size)
        else:
            result[label] = render_blank(label, anchor=anchor, size=size)
    return result


def empirical_mutual_information_binary(xs: Sequence[int], ys: Sequence[int]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Need equal non-empty sequences")
    n = len(xs)
    joint = {(x, y): 0 for x in (0, 1) for y in (0, 1)}
    px = {0: 0, 1: 0}
    py = {0: 0, 1: 0}
    for x, y in zip(xs, ys):
        joint[(x, y)] += 1
        px[x] += 1
        py[y] += 1
    mi = 0.0
    for (x, y), c in joint.items():
        if c == 0:
            continue
        pxy = c / n
        mi += pxy * math.log2(pxy / ((px[x]/n) * (py[y]/n)))
    return mi


def powerset(coalition: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    c = tuple(sorted(coalition))
    for r in range(len(c) + 1):
        yield from itertools.combinations(c, r)


def canonical_subset(items: Iterable[str]) -> str:
    s = "".join(sorted(items))
    return "EMPTY" if not s else s


def harsanyi_dividend(values: Mapping[str, float], coalition: Sequence[str]) -> float:
    coalition = tuple(sorted(coalition))
    total = 0.0
    k = len(coalition)
    for subset in powerset(coalition):
        name = canonical_subset(subset)
        total += ((-1) ** (k - len(subset))) * values[name]
    return total


def leave_one_out(values: Mapping[str, float], item: str, full: str = "ABCD") -> float:
    full_set = set(full)
    if item not in full_set:
        raise ValueError(item)
    reduced = canonical_subset(full_set - {item})
    return values[canonical_subset(full_set)] - values[reduced]


def wrong_pairs() -> Tuple[Tuple[str, str], ...]:
    return (("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"))


def correct_pairs() -> Tuple[Tuple[str, str], ...]:
    return (("A", "B"), ("C", "D"))


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = 17,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha/2) * (n_boot-1))]
    hi = means[int((1-alpha/2) * (n_boot-1))]
    return lo, hi
