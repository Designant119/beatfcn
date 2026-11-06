import math
import unittest

import torch

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.hn2_native import HN2Config, HN2Native
from beatfcn.train.hn2_native import (
    EndpointDistribution, canonical_targets, generate, loss_pass, prepare_public_support,
    two_pass, validate_targets,
)
from beatfcn.train.native_diagnostic_priors import fit_priors, duration_probability


def narrow_config(d=64, max_duration_us=100_000):
    return HN2Config(d=d, heads=4, ffn=4*d, fine=max(16, d//2),
                     max_duration_us=max_duration_us)


def toy_batch(request=None, regions=()):
    request = request or EventRequest('phira', 0, 8000, 12000,
                                     copy_times=(1501, 1502, 7001), max_events=512, max_tokens=1537)
    g = torch.Generator().manual_seed(34)
    x = dict(local=torch.randn(1, 2, 8, 1, 1024, generator=g)*.1,
             local_times=(torch.arange(8)[None]+.5)*.04,
             local_valid=torch.ones(1, 8, dtype=torch.bool),
             mel=torch.randn(1, 32, 128, generator=g)*.1,
             mel_times=(torch.arange(32)[None]+.5)*.01,
             mel_valid=torch.ones(1, 32, dtype=torch.bool),
             song=torch.randn(1, 8, 1024, generator=g)*.1,
             song_times=(torch.arange(8)[None]+.5)*.5,
             song_valid=torch.ones(1, 8, dtype=torch.bool),
             reference=torch.zeros(1, 3, 16), reference_times=torch.zeros(1, 3),
             reference_valid=torch.tensor([[True, False, False]]),
             controls=torch.zeros(1, 8, 6), mode=torch.ones(1, dtype=torch.long))
    x['whole_mel'], x['whole_mel_times'], x['whole_mel_valid'] = x['mel'].clone(), x['mel_times'].clone(), x['mel_valid'].clone()
    x.update(prepare_public_support([request], [regions], max_duration_us=100_000))
    targets = torch.zeros(1, 512, 4, dtype=torch.int64)
    valid = torch.zeros(1, 512, dtype=torch.bool)
    return dict(inputs=x, targets=targets, target_valid=valid), request


class HN2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_actual_meta_capacity_and_narrow_family(self):
        with torch.device('meta'):
            model = HN2Native()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 138621056)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(narrow_config(128).slots, 512)

    def test_public_support_fractional_duplicates_known_regions_and_boundary(self):
        batch, request = toy_batch(regions=((2000, 3000),))
        x = batch['inputs']
        heads = x['head_us'][0, x['head_valid'][0]].tolist()
        self.assertIn(1501, heads)
        self.assertNotIn(2000, heads)
        self.assertNotIn(8000, heads)
        rows = [[1000, 1000, 0, -1], [1501, 7001, 1, -1], [1501, 7001, 1, -1],
                [3000, 3000, 2, -1], [3000, 3000, 3, -1]]
        batch['targets'][0, :5] = torch.tensor(rows)
        batch['target_valid'][0, :5] = True
        self.assertTrue(validate_targets(x, batch['targets'], batch['target_valid']))
        batch['targets'][0, 0, 0] = 2000
        with self.assertRaises(ValueError):
            validate_targets(x, batch['targets'], batch['target_valid'])

    def test_multiblock_cross_core_endpoint_boundaries_and_allowed_types(self):
        request = EventRequest('phira', 1000, 5000, 4_123_456,
                               copy_times=(1501, 1502, 2_040_001, 4_123_456), allowed_types=('hold',),
                               max_events=512, max_tokens=8192)
        inputs = prepare_public_support([request], max_duration_us=5_000_000)
        self.assertTrue(inputs['type_allowed'][..., 1].all())
        self.assertFalse(inputs['type_allowed'][..., 0].any())
        prior = fit_priors([], [], [1, 1, 1, 1])
        heads = torch.full((1, 512), 1501, dtype=torch.int64)
        tree = EndpointDistribution(inputs, heads, prior, 5_000_000)
        class ZeroModel:
            def tail_scores(self, output, heads, centers, level):
                return torch.zeros_like(centers, dtype=torch.float32)
        output = dict(states=torch.zeros(1, 512, 1))
        for endpoint in (1502, 2000, 40000, 41000, 2_000_000, 2_001_000, 2_040_001, 4_123_456):
            end, nll, supported, _ = tree.factors(ZeroModel(), output, target=torch.full_like(heads, endpoint))
            self.assertTrue(bool(supported.all()))
            self.assertTrue((end == endpoint).all())
            expected = duration_probability(prior, 1501, endpoint, request.duration_us, request.copy_times)
            self.assertAlmostEqual(float(nll[0, 0]), -math.log(expected), places=5)
        rows = torch.zeros(1, 512, 4, dtype=torch.int64)
        rows[0, :3] = torch.tensor([[2000, 2000, 3, -1], [1000, 6000, 1, -1], [2000, 2000, 0, -1]])
        good = torch.zeros(1, 512, dtype=torch.bool); good[:, :3] = True
        ordered, flags = canonical_targets(rows, good)
        self.assertEqual(ordered[0, :3, 2].tolist(), [1, 0, 3])
        self.assertEqual(int(flags.sum()), 3)

    def test_exact_endpoint_all_leaf_probability_matches_scalar(self):
        batch, request = toy_batch()
        batch['inputs']['copy_us'] = batch['inputs']['copy_us'].repeat(1, 2)
        batch['inputs']['copy_valid'] = batch['inputs']['copy_valid'].repeat(1, 2)
        model = HN2Native(narrow_config())
        prior = fit_priors([], [], [1, 1, 1, 1])
        output = model(batch['inputs'])
        h = 1501
        legal = sorted(set(range(2000, request.duration_us+1, 1000)) | {1502, 7001})
        heads = torch.full((1, 512), h, dtype=torch.int64)
        tree = EndpointDistribution(batch['inputs'], heads, prior, 100_000)
        actual = []
        for end in legal:
            target = torch.full_like(heads, end)
            _, nll, supported, records = tree.factors(model, output, target=target)
            self.assertTrue(bool(supported.all()))
            expected = duration_probability(prior, h, end, request.duration_us, request.copy_times)
            self.assertLess(abs(float(nll[0, 0].detach())+math.log(expected)), 2e-6)
            for r in records:
                self.assertLess(abs(float(r['logits'][0, 0].detach().softmax(-1).sum())-1), 2e-6)
            actual.append(math.exp(-float(nll[0, 0].detach())))
        self.assertAlmostEqual(sum(actual), 1, places=6)

    def test_two_pass_gradients_and_label_isolation(self):
        torch.manual_seed(878)
        batch, _ = toy_batch()
        rows = [[1000, 1000, 0, -1], [1501, 7001, 1, -1], [1501, 7001, 1, -1],
                [2000, 2000, 2, -1], [2000, 2000, 3, -1]]
        batch['targets'][0, :5] = torch.tensor(rows)
        batch['target_valid'][0, :5] = True
        model = HN2Native(narrow_config())
        prior = fit_priors([], [], [1, 1, 1, 1])
        rng = torch.Generator().manual_seed(771)
        loss, metrics, z, y = two_pass(model, batch['inputs'], batch['targets'], batch['target_valid'], prior, generator=rng)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(y)
        self.assertFalse(z['events'].requires_grad)
        self.assertTrue(bool(metrics['p0']['supported']))
        loss.backward()
        names = dict(model.named_parameters())
        for stem in ('local_layers.0', 'global_layers.0', 'relation_layers.0', 'decoder.0',
                     'whole_projection', 'type_read', 'tail_residual'):
            gradients = [p.grad for n,p in names.items() if n.startswith(stem) and p.grad is not None]
            self.assertTrue(gradients, stem)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), stem)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0, stem)
        # Changing hidden labels cannot change public forward inputs or q0.
        a = model(batch['inputs'])['active_logits'].detach()
        batch['targets'].fill_(999)
        b = model(batch['inputs'])['active_logits'].detach()
        self.assertTrue(torch.equal(a, b))

    def test_cpu_mps_fixed_machine_state_probabilities_and_gradients(self):
        if torch.__version__ != '2.14.0' or not torch.backends.mps.is_available():
            self.skipTest('Requires installed isolated2.14 MPS; CPU tests remain applicable')
        import os
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            self.fail('Silent MPS fallback prohibited')
        torch.mps.set_per_process_memory_fraction(.5)
        batch, _ = toy_batch()
        rows = torch.tensor([[1000, 1000, 0, -1], [1501, 11000, 1, -1],
                             [2000, 2000, 2, -1], [2000, 2000, 3, -1]])
        batch['targets'][0, :4] = rows
        batch['target_valid'][0, :4] = True
        state = dict(events=torch.zeros_like(batch['targets']), active=torch.zeros_like(batch['target_valid']))
        state['events'][0, :4] = rows
        state['active'][0, :4] = True
        torch.manual_seed(878)
        base_model = HN2Native(narrow_config())
        with torch.no_grad():
            base_model.tail_residual.weight.fill_(.01)
        base = {n:p.detach().clone() for n,p in base_model.state_dict().items()}
        prior = fit_priors([], [], [1, 1, 1, 1])
        results = []
        for device in ('cpu', 'mps'):
            model = HN2Native(narrow_config()).to(device)
            model.load_state_dict(base)
            inputs = {k:v.to(device) for k,v in batch['inputs'].items()}
            out = model(inputs, state={k:v.to(device) for k,v in state.items()}, pass_index=1)
            loss, details = loss_pass(model, out, batch['targets'].to(device), batch['target_valid'].to(device), prior, details=True)
            loss.backward()
            results.append(dict(loss=float(loss.detach().cpu()),
                head=model.head_logits(out).detach().cpu(), type=details['type_logits'].cpu(),
                tail=[r['logits'][r['valid']].cpu() for r in details['tail_factors']],
                gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}))
            del out, model, inputs, loss, details
        a, b = results
        self.assertLess(abs(a['loss']-b['loss']), 1e-4)
        for name in ('head', 'type'):
            self.assertLess(float((a[name]-b[name]).abs().max()), 1e-4, name)
        for i,(left,right) in enumerate(zip(a['tail'],b['tail'])):
            self.assertLess(float((left-right).abs().max()), 1e-4, f'tail{i}')
        self.assertEqual(set(a['gradients']), set(b['gradients']))
        for name,left in a['gradients'].items():
            right = b['gradients'][name]
            self.assertTrue(torch.isfinite(right).all(), name)
            self.assertLess(float((left-right).abs().max()), 1e-4, name)
        torch.mps.empty_cache()

    def test_forced_null_generate_replay_and_four_allowed_types(self):
        prior = fit_priors([], [], [1, 1, 1, 1])
        batch, _ = toy_batch(regions=((0, 8000),))
        model = HN2Native(narrow_config())
        loss, metrics, z, y = two_pass(model, batch['inputs'], batch['targets'], batch['target_valid'], prior,
                                      generator=torch.Generator().manual_seed(44), sample_final=True)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(int(z['active'].sum()), 0)
        self.assertEqual(int(y['active'].sum()), 0)
        batch, request = toy_batch()
        z1, y1 = generate(model, batch['inputs'], prior, generator=torch.Generator().manual_seed(44))
        z2, y2 = generate(model, batch['inputs'], prior, generator=torch.Generator().manual_seed(44))
        self.assertTrue(torch.equal(y1['events'], y2['events']))
        self.assertTrue(torch.equal(y1['active'], y2['active']))
        self.assertTrue(validate_targets(batch['inputs'], y1['events'], y1['active']))
        self.assertEqual(len(set(y1['events'][0, y1['active'][0], 2].tolist())), 4)


if __name__ == '__main__':
    unittest.main()
