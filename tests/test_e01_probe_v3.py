import math

from e01_probe_v3.probe_core import (
    LABELS,
    SUBSET_NAMES,
    build_panels,
    generate_coalition_samples,
    harsanyi_dividend,
    leave_one_out,
    wrong_pairs,
)


def ideal_redundant_value(subset: str) -> float:
    s = set() if subset == "EMPTY" else set(subset)
    # AB or CD is sufficient; otherwise the binary target is maximally uncertain.
    return 0.0 if ({"A", "B"} <= s or {"C", "D"} <= s) else -math.log(2.0)


def test_lattice_has_all_16_subsets():
    assert len(SUBSET_NAMES) == 16
    assert len(set(SUBSET_NAMES)) == 16
    assert "EMPTY" in SUBSET_NAMES
    assert "ABCD" in SUBSET_NAMES


def test_redundant_coalition_generator_obeys_rule():
    rows = generate_coalition_samples(64, "xor")
    for row in rows:
        a, b, c, d = (row.bits[x] for x in LABELS)
        assert (a ^ b) == row.target
        assert (c ^ d) == row.target


def test_oracle_loo_is_zero_under_redundancy():
    values = {s: ideal_redundant_value(s) for s in SUBSET_NAMES}
    for label in LABELS:
        assert abs(leave_one_out(values, label)) < 1e-12


def test_oracle_pair_dividends():
    values = {s: ideal_redundant_value(s) for s in SUBSET_NAMES}
    assert math.isclose(
        harsanyi_dividend(values, ("A", "B")), math.log(2.0), abs_tol=1e-12
    )
    assert math.isclose(
        harsanyi_dividend(values, ("C", "D")), math.log(2.0), abs_tol=1e-12
    )
    for pair in wrong_pairs():
        assert abs(harsanyi_dividend(values, pair)) < 1e-12


def test_fixed_slot_builder_always_returns_four_slots():
    row = generate_coalition_samples(1, "xor")[0]
    for subset in SUBSET_NAMES:
        panels = build_panels(row.bits, row.styles, anchor=True, subset=subset)
        assert tuple(panels.keys()) == LABELS
        assert len(panels) == 4
        assert len({im.size for im in panels.values()}) == 1
