import math
import unittest

from e01_probe.build_dataset_v2 import build_samples
from e01_probe.report import summarize
from process.interaction_credit import (
    canonical, harsanyi_dividend, higher_order_interaction, leave_one_out,
    pair_interaction, standalone_marginal, subsets,
)


def oracle_values(population, row, target):
    """Enumerate the conditional finite population, never encode desired dividends."""
    values = {}
    for subset in subsets():
        labels = '' if subset == 'EMPTY' else subset
        matches = [r for r in population if all(r['bits'][i] == row['bits'][i] for i in labels)]
        correct = sum(r['targets'][target] == row['targets'][target] for r in matches)
        values[subset] = math.log(correct / len(matches))
    return values


def oracle_records():
    raw, controls = [], []
    for probe in ('A', 'B'):
        population = build_samples(probe, 32)
        for row in population:
            for target, truth in row['targets'].items():
                values = oracle_values(population, row, target)
                for subset, value in values.items():
                    raw.append({'probe': probe, 'sample_id': row['sample_id'],
                                'target_field': target, 'subset': subset, 'ground_truth': truth,
                                f'logp_{truth}': value, 'correct': 1})
                controls.append(dict(probe=probe, sample_id=row['sample_id'], target_field=target,
                                     control='text_only_logic', correct=1))
            for label in 'ABCD':
                controls.append(dict(probe=probe, sample_id=row['sample_id'], target_field=label,
                                     control='single_cue', correct=1))
    return raw, controls


class InteractionTest(unittest.TestCase):
    def test_ideal_bayes_a(self):
        population = build_samples('A')
        for row in population:
            for target, good in [('STEP1', 'AB'), ('STEP2', 'CD'), ('FINAL', 'ABCD')]:
                values = oracle_values(population, row, target)
                self.assertAlmostEqual(higher_order_interaction(values, good), math.log(2))
                for subset in subsets()[1:]:
                    self.assertAlmostEqual(harsanyi_dividend(values, subset),
                                           math.log(2) if subset == good else 0)

    def test_ideal_bayes_b(self):
        population = build_samples('B')
        for row in population:
            values = oracle_values(population, row, 'STEP')
            for i in 'ABCD':
                self.assertAlmostEqual(standalone_marginal(values, i), 0)
                self.assertAlmostEqual(leave_one_out(values, i), 0)
            for pair in [s for s in subsets() if len(s) == 2]:
                self.assertAlmostEqual(pair_interaction(values, *pair),
                                       math.log(2) if pair in ('AB', 'CD') else 0)

    def test_mobius_reconstruction(self):
        values = {s: (n * n - 3 * n) / 7 for n, s in enumerate(subsets())}
        for s in subsets():
            self.assertAlmostEqual(sum(harsanyi_dividend(values, t) for t in subsets(s)), values[s])
        self.assertEqual(canonical('DB'), 'BD')
        with self.assertRaises(ValueError):
            canonical('AA')
        with self.assertRaises(KeyError):
            harsanyi_dividend({'EMPTY': 0}, 'AB')

    def test_oracle_report_and_gates(self):
        raw, controls = oracle_records()
        report, credits = summarize(raw, controls, draws=200)
        self.assertEqual(set(report['gates'].values()), {'PASS'})
        self.assertIsNone(report['STOP_REASON'])
        self.assertEqual(len(credits), 32 * 4 * 19)
        self.assertAlmostEqual(report['probe_A']['final_four_way_interaction']['mean'], math.log(2))
        report_a, _ = summarize([r for r in raw if r['probe'] == 'A'],
                                [r for r in controls if r['probe'] == 'A'], draws=200)
        self.assertEqual(report_a['gates']['G4_LOO_SEPARATION'], 'NOT_RUN')

    def test_failure_reasons_and_equivalence(self):
        raw, controls = oracle_records()
        for row in controls:
            if row['control'] == 'text_only_logic':
                row['correct'] = 0
        report, _ = summarize(raw, controls, draws=200)
        self.assertEqual(report['STOP_REASON'], 'LOGIC_FAILURE')
        raw, controls = oracle_records()
        for row in raw:
            if row['probe'] == 'B' and len(row['subset']) == 3:
                row[f"logp_{row['ground_truth']}"] -= .2
        report, _ = summarize(raw, controls, draws=200)
        self.assertIn('PROBE_LEAKAGE', report['stop_reasons'])
        raw, controls = oracle_records()
        for row in raw:
            if row['probe'] == 'B' and len(row['subset']) == 3:
                row[f"logp_{row['ground_truth']}"] -= .4 if int(row['sample_id'][-4:]) % 2 else -.4
        report, _ = summarize(raw, controls, draws=200)
        self.assertEqual(report['gates']['G4_LOO_SEPARATION'], 'INCONCLUSIVE')

    def test_incomplete_or_duplicate_scores_fail(self):
        raw, controls = oracle_records()
        with self.assertRaises(ValueError):
            summarize(raw[1:], controls)
        with self.assertRaises(ValueError):
            summarize(raw + raw[:1], controls)


if __name__ == '__main__':
    unittest.main()
