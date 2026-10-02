import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from e01_probe.build_dataset_v2 import build_samples, render_cue, validate, write_dataset
from e01_probe.run import question
from models.vlm.qwen_vl_predict import build_multi_image_prompt, score_binary_candidates_qwen


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class FakeProcessor:
    def __init__(self, multi=False, no_text=False, error=None):
        self.tokenizer = self
        self.multi, self.no_text, self.error = multi, no_text, error
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        return {'0': [1, 3], '1': [2, 4]}[text] if self.multi else {'0': [1], '1': [2]}[text]

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'prompt'

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise ValueError(self.error)
        if self.no_text and 'images' not in kwargs:
            raise ValueError('Images required; got None')
        return Batch(input_ids=torch.tensor([[0, 5, 0]]), attention_mask=torch.ones(1, 3, dtype=torch.long))


class FakeModel:
    device = 'cpu'

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs['input_ids']
        logits = torch.zeros(1, ids.shape[1], 6)
        logits[..., 1] = 1
        logits[..., 2] = 2
        # Distinct logits at the second candidate token position.
        if ids.shape[1] > 3:
            logits[:, 3, 3] = 3
            logits[:, 3, 4] = -1
        return SimpleNamespace(logits=logits)


class GeneratorTest(unittest.TestCase):
    def test_balance_and_reproducibility(self):
        for probe in ('A', 'B'):
            for count in (32, 128):
                rows = build_samples(probe, count)
                validate(rows, probe)
                self.assertEqual(rows, build_samples(probe, count))
                self.assertEqual(rows, build_samples(probe, 128)[:count])
                for row in rows:
                    self.assertEqual(len(set(row['cue_styles'].values())), 4)
            for label in 'ABCD':
                self.assertEqual({r['cue_styles'][label] for r in build_samples(probe)},
                                 {'position', 'color', 'size', 'orientation'})
            with self.assertRaises(ValueError):
                build_samples(probe, 31)
            self.assertNotEqual(build_samples(probe), build_samples(probe, seed=43))

    def test_images_and_manifest(self):
        rows = build_samples('A', 32)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'dataset'
            write_dataset(rows, directory)
            self.assertEqual(len(list(directory.glob('*.png'))), 128)
            self.assertEqual(json.loads((directory / 'manifest.json').read_text()), rows)
        for style in rows[0]['cue_styles'].values():
            a, b = render_cue(0, style), render_cue(1, style)
            self.assertEqual(a.size, (224, 224))
            self.assertNotEqual(a.tobytes(), b.tobytes())

    def test_prompt_order_and_no_truth_leak(self):
        image = render_cue(0, 'size')
        messages, selected = build_multi_image_prompt('What is STEP?', {'B': image, 'D': image}, 'DB')
        self.assertEqual([x['text'] for x in messages[0]['content'] if x['type'] == 'text'][:2],
                         ['Image B:', 'Image D:'])
        self.assertEqual(len(selected), 2)
        self.assertEqual(build_multi_image_prompt('?', {}, 'EMPTY')[1], [])
        rows = build_samples('A', 32)
        self.assertEqual(question(rows[0], 'STEP1'), question(rows[1], 'STEP1'))
        self.assertIn('A=', question(rows[0], 'STEP1', True))

    def test_next_token_scoring(self):
        model = FakeModel()
        result = score_binary_candidates_qwen(model, FakeProcessor(), '?', {}, 'EMPTY')
        self.assertAlmostEqual(result['p_0'] + result['p_1'], 1, places=6)
        self.assertAlmostEqual(result['logp_1'] - result['logp_0'], 1, places=6)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result['empty_mode'], 'text_only')
        self.assertEqual(result['scoring_mode'], 'next_token')

    def test_teacher_forcing_sums_only_answer(self):
        model = FakeModel()
        result = score_binary_candidates_qwen(model, FakeProcessor(multi=True), '?', {}, 'EMPTY')
        first = torch.log_softmax(torch.tensor([0., 1., 2., 0., 0., 0.]), 0)
        second = torch.log_softmax(torch.tensor([0., 1., 2., 3., -1., 0.]), 0)
        self.assertAlmostEqual(result['candidate_loglik_0'], (first[1] + second[3]).item(), places=6)
        self.assertAlmostEqual(result['candidate_loglik_1'], (first[2] + second[4]).item(), places=6)
        self.assertEqual(model.calls[0]['input_ids'].tolist(), [[0, 5, 0, 1, 3]])
        self.assertEqual(result['scoring_mode'], 'teacher_forcing')
        self.assertEqual(tuple(model.calls[0]['attention_mask'].shape), (1, 5))

    def test_runner_writes_complete_lattices_and_controls(self):
        from e01_probe.run import main
        model = FakeModel()
        model.config = SimpleNamespace(_commit_hash='fake-test-model')
        model.dtype = torch.float32
        model.eval = lambda: model
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            argv = ['probe_e01_v2.py', '--probe', 'both', '--limit', '32',
                    '--device', 'cpu', '--bootstrap-draws', '100', '--output-dir', str(output)]
            with patch('sys.argv', argv), patch('models.vlm.loader.load_qwen25_vl',
                                               return_value=(model, FakeProcessor())), \
                    patch('importlib.metadata.version', return_value='test'), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            import csv
            with (output / 'raw_scores.csv').open() as stream:
                raw = list(csv.DictReader(stream))
            with (output / 'controls.csv').open() as stream:
                controls = list(csv.DictReader(stream))
            self.assertEqual(len(raw), 2048)
            self.assertEqual(len(controls), 384)
            self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'COMPLETE')
            self.assertTrue((output / 'summary.json').is_file())
            with patch('sys.argv', argv), self.assertRaises(FileExistsError):
                main()

    def test_blank_fallback_is_recorded_and_narrow(self):
        result = score_binary_candidates_qwen(FakeModel(), FakeProcessor(no_text=True), '?', {}, 'EMPTY')
        self.assertEqual(result['empty_mode'], 'blank_image')
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            score_binary_candidates_qwen(FakeModel(), FakeProcessor(error='unrelated'), '?', {}, 'EMPTY')


if __name__ == '__main__':
    unittest.main()
