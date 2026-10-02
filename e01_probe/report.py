"""Sample-level paired bootstrap; all log-probability effects are in nats."""
from collections import defaultdict
import itertools
import numpy as np

from process.interaction_credit import harsanyi_dividend, leave_one_out, subsets

PAIRS = [s for s in subsets() if len(s) == 2]


def bootstrap(values, seed=42, draws=2000):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Bootstrap needs finite, nonempty values')
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return {'mean': float(x.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'n_samples': len(x)}


def summarize(raw, controls, seed=42, draws=2000, ability_threshold=.8, loo_margin=.05):
    grouped = defaultdict(dict)
    for row in raw:
        key = (row['probe'], row['sample_id'], row['target_field'])
        if row['subset'] in grouped[key]:
            raise ValueError('Duplicate intervention')
        grouped[key][row['subset']] = row
    metrics = defaultdict(list)
    credits = []
    for (probe, sample_id, target), rows in grouped.items():
        if set(rows) != set(subsets()):
            raise ValueError('Incomplete subset lattice')
        values = {s: float(r[f"logp_{int(r['ground_truth'])}"]) for s, r in rows.items()}
        gamma = {s: harsanyi_dividend(values, s) for s in subsets()[1:]}
        loo = {i: leave_one_out(values, i) for i in 'ABCD'}
        for s, effect in gamma.items():
            credits.append(dict(probe=probe, sample_id=sample_id, target_field=target,
                                metric='harsanyi', coalition=s, value=effect))
        for i, effect in loo.items():
            credits.append(dict(probe=probe, sample_id=sample_id, target_field=target,
                                metric='LOO', coalition=i, value=effect))
        prefix = f'{probe}.{target}'
        metrics[f'{prefix}.full_accuracy'].append(int(rows['ABCD']['correct']))
        for s, effect in gamma.items():
            metrics[f'{prefix}.gamma_{s}'].append(effect)
        for i, effect in loo.items():
            metrics[f'{prefix}.LOO_{i}'].append(effect)
        if probe == 'A' and target in ('STEP1', 'STEP2'):
            good = 'AB' if target == 'STEP1' else 'CD'
            for wrong in set(PAIRS) - {good}:
                metrics[f'{prefix}.contrast_{good}_{wrong}'].append(gamma[good] - gamma[wrong])
        if probe == 'A' and target == 'FINAL':
            for lower in subsets()[1:-1]:
                metrics[f'{prefix}.contrast_ABCD_{lower}'].append(gamma['ABCD'] - gamma[lower])
        if probe == 'B':
            for good, wrong in itertools.product(('AB', 'CD'), ('AC', 'AD', 'BC', 'BD')):
                metrics[f'{prefix}.contrast_{good}_{wrong}'].append(gamma[good] - gamma[wrong])
    # Bootstrap each control question across samples, not pooled correlated questions.
    for row in controls:
        metrics[f"{row['probe']}.{row['control']}.{row['target_field']}"].append(int(row['correct']))
    stats = {k: bootstrap(v, seed, draws) for k, v in sorted(metrics.items())}
    probes = sorted({row['probe'] for row in raw})
    full_accuracy = {k: v for k, v in stats.items() if k.endswith('.full_accuracy')}
    logic = {k: v for k, v in stats.items() if '.text_only_logic.' in k}
    perception = {k: v for k, v in stats.items() if '.single_cue.' in k}

    def positive(keys):
        if not keys:
            return 'NOT_RUN'
        if all(stats[k]['ci95'][0] > 0 for k in keys):
            return 'PASS'
        if any(stats[k]['ci95'][1] <= 0 for k in keys):
            return 'FAIL'
        return 'INCONCLUSIVE'

    def ability(records):
        if not records:
            return 'NOT_RUN'
        if all(v['ci95'][0] >= ability_threshold for v in records.values()):
            return 'PASS'
        if any(v['ci95'][1] < ability_threshold for v in records.values()):
            return 'FAIL'
        return 'INCONCLUSIVE'

    def combine(states):
        for state in ('FAIL', 'INCONCLUSIVE', 'NOT_RUN'):
            if state in states:
                return state
        return 'PASS'

    logic_gate, perception_gate = ability(logic), ability(perception)
    g1 = combine([logic_gate, perception_gate, ability(full_accuracy)])
    step_keys = [k for k in stats if k.startswith(('A.STEP1.contrast_', 'A.STEP2.contrast_'))]
    step_keys += [k for k in ('A.STEP1.gamma_AB', 'A.STEP2.gamma_CD') if k in stats]
    g2 = positive(step_keys)
    final_keys = [k for k in stats if k.startswith('A.FINAL.contrast_')]
    final_keys += [k for k in ('A.FINAL.gamma_ABCD',) if k in stats]
    g3 = positive(final_keys)
    b_loo = {i: stats[f'B.STEP.LOO_{i}'] for i in 'ABCD'} if 'B' in probes else {}
    if not b_loo:
        loo_gate = 'NOT_RUN'
    elif all(-loo_margin <= v['ci95'][0] and v['ci95'][1] <= loo_margin for v in b_loo.values()):
        loo_gate = 'PASS'
    elif any(v['ci95'][0] > loo_margin or v['ci95'][1] < -loo_margin for v in b_loo.values()):
        loo_gate = 'FAIL'
    else:
        loo_gate = 'INCONCLUSIVE'
    b_keys = [k for k in stats if k.startswith('B.STEP.contrast_')]
    b_keys += [k for k in ('B.STEP.gamma_AB', 'B.STEP.gamma_CD') if k in stats]
    coalition_gate = positive(b_keys)
    wrong = {p: stats[f'B.STEP.gamma_{p}'] for p in ('AC', 'AD', 'BC', 'BD')} if 'B' in probes else {}
    wrong_gate = ('PASS' if wrong and all(-loo_margin <= v['ci95'][0] and v['ci95'][1] <= loo_margin
                                        for v in wrong.values()) else 'INCONCLUSIVE' if wrong else 'NOT_RUN')
    g4 = combine([loo_gate, coalition_gate, wrong_gate])
    reasons = []
    for state, reason in [(logic_gate, 'LOGIC_FAILURE'), (perception_gate, 'PERCEPTION_FAILURE'),
                          (g2, 'NO_STEP_INTERACTION'), (loo_gate, 'PROBE_LEAKAGE'),
                          (coalition_gate, 'NO_COALITION_SIGNAL')]:
        if state == 'FAIL':
            reasons.append(reason)
    gates = dict(G1_BASIC_ABILITY=g1, G2_STEP_IDENTIFIABILITY=g2,
                 G3_FINAL_ONLY_SEPARATION=g3, G4_LOO_SEPARATION=g4)
    report = {
        'full_evidence_accuracy': full_accuracy, 'text_only_logic_accuracy': logic,
        'single_cue_accuracy': perception,
        'probe_A': {
            'step1_correct_pair_interaction': stats.get('A.STEP1.gamma_AB'),
            'step2_correct_pair_interaction': stats.get('A.STEP2.gamma_CD'),
            'final_four_way_interaction': stats.get('A.FINAL.gamma_ABCD'),
            'wrong_pair_interactions': {k: v for k, v in stats.items()
                                       if k.startswith(('A.STEP1.gamma_', 'A.STEP2.gamma_', 'A.FINAL.gamma_'))
                                       and len(k.rsplit('_', 1)[1]) == 2
                                       and k not in ('A.STEP1.gamma_AB', 'A.STEP2.gamma_CD')},
            'step_conditioned_LOO': {k: v for k, v in stats.items() if k.startswith('A.') and '.LOO_' in k}},
        'probe_B': {'mean_LOO': b_loo, 'gamma_AB': stats.get('B.STEP.gamma_AB'),
                    'gamma_CD': stats.get('B.STEP.gamma_CD'), 'wrong_pair_gamma': wrong},
        'gates': gates, 'STOP_REASON': reasons[0] if reasons else None, 'stop_reasons': reasons,
        'overall_status': combine(list(gates.values())), 'all_metrics': stats,
        'thresholds': {'ability_accuracy': ability_threshold, 'equivalence_margin_nats': loo_margin,
                       'positive_effect_ci_lower': 0},
        'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'sample within probe and target',
                      'interval': 'pointwise percentile 95%; contrasts paired within sample',
                      'scope': 'synthetic sample variability; not model/training-seed uncertainty'},
    }
    return report, credits
