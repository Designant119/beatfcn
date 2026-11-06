from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.local_group_diffusion import (
    MASK, GroupDiffusionCanvas, GroupSampleError, clean_packages, corrupt_packages,
    group_diffusion_loss, sample_group_diffusion, valid_groups,
)
from beatfcn.train.local_typed_canvas import encode_canvas, request_layout
from tests import test_local_event_package as package_tests
from tests.test_local_event_package import RejectTargetReads
from tests.test_local_mamba_canvas import small_config


class GroupDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8001)
        self.request, self.inputs, self.target = package_tests.EventPackageTests.synthetic_example()
        self.model = GroupDiffusionCanvas(small_config(), density_reference=4, diffusion_steps=4).eval()
        self.groups = valid_groups(self.inputs, self.request)
        self.clean = clean_packages(self.target, self.request)
        self.steps = torch.tensor([4])
        self.noisy = corrupt_packages(self.clean, self.groups, self.steps, 4)

    def test_mask_hides_entire_count_and_members_not_sixteen_absent_slots(self):
        self.assertTrue((self.noisy['count'][self.groups] == MASK).all())
        self.assertEqual(int(self.noisy['offset'].abs().sum()), 0)
        self.assertEqual(int(self.noisy['tail'].abs().sum()), 0)
        different = {key: value.clone() for key, value in self.clean.items()}
        different['count'][self.groups] = 16
        different['offset'].fill_(39)
        different['tail'].fill_(3000)
        hidden = corrupt_packages(different, self.groups, self.steps, 4)
        for key in hidden:
            torch.testing.assert_close(hidden[key], self.noisy[key], atol=0, rtol=0)
        # Even stale/poisoned member fields inside MASK cannot enter embeddings.
        poison = {key: value.clone() for key, value in self.noisy.items()}
        poison['offset'][self.groups] = 99999
        poison['tail'][self.groups] = -99999
        torch.testing.assert_close(self.model.package_features(poison),
                                   self.model.package_features(self.noisy), atol=0, rtol=0)

    def test_observed_neighbor_changes_masked_group_probability(self):
        observed = {key: value.clone() for key, value in self.noisy.items()}
        observed['count'][0, 0, 0] = 1
        observed['offset'][0, 0, 0, 0] = 4
        audio = self.model.encode_audio(self.inputs)
        baseline = self.model(self.inputs, self.noisy, self.steps, audio=audio)
        changed = self.model(self.inputs, observed, self.steps, audio=audio)
        self.assertGreater(float((baseline['count'][0, 1] - changed['count'][0, 1]).abs().max()), 1e-7)
        torch.testing.assert_close(changed['count'].sum(-1), torch.ones(1, 200, 4))

    def test_density_is_one_exponential_tilt_on_all_seventeen_counts(self):
        changed = {key: value.clone() for key, value in self.inputs.items()}
        changed['controls'][0, 0, :3] = torch.tensor([8 / 64, 10 / 64, 1.])
        first_audio = self.model.encode_audio(self.inputs)
        second_audio = self.model.encode_audio(changed)
        torch.testing.assert_close(first_audio, second_audio, atol=0, rtol=0)
        first = self.model(self.inputs, self.noisy, self.steps, audio=first_audio)
        second = self.model(changed, self.noisy, self.steps, audio=first_audio)
        tilt = F.softplus(self.model.density_alpha) * torch.tensor(9 / 4).log()
        expected = (first['count_log_probs'] + tilt * torch.arange(17)).softmax(-1)
        torch.testing.assert_close(second['count'], expected, atol=1e-6, rtol=1e-6)

    def test_common_package_likelihood_reaches_audio_count_members_and_tails(self):
        loss, parts = group_diffusion_loss(self.model, self.inputs, self.target, self.request,
                                           self.steps, noisy=self.noisy)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, parts['count'] + parts['onset'] + parts['tail'])
        self.assertEqual(int(parts['valid_groups']), 8)
        self.assertEqual(int(parts['masked_groups']), 8)
        loss.backward()
        parameters = dict(self.model.named_parameters())
        for name in ('frame_read.weight', 'count_gate.weight', 'member_gru.weight_ih',
                     'onset_read.weight', 'tail_block.weight', 'noise_time.weight', 'mask_package'):
            self.assertIsNotNone(parameters[name].grad, name)
            self.assertTrue(torch.isfinite(parameters[name].grad).all(), name)
            self.assertGreater(float(parameters[name].grad.abs().sum()), 0, name)

    def copy_example(self):
        request = replace(self.request, copy_times=(12_345,))
        inputs = {key: value.clone() for key, value in self.inputs.items()}
        inputs.update({key: torch.from_numpy(np.asarray(value))[None]
                       for key, value in request_layout(request).items()})
        return request, inputs

    def test_ordered_hold_support_preserves_microsecond_heads_native_tails(self):
        request, inputs = self.copy_example()
        address = int((inputs['canvas_head_us'][0, 0] == 12_345).nonzero()[0])
        with torch.no_grad():
            self.model.onset_read.weight.zero_()
            self.model.onset_read.bias.fill_(-10000)
            self.model.onset_read.bias[address] = 10000
            x = self.model.encode_audio(inputs)[0]
            members, _, _ = self.model.ordered_members(x, 0, 1, 3, inputs, request,
                                                      rng=np.random.default_rng(54))
        self.assertEqual([int(inputs['canvas_head_us'][0, 0, o]) for o, _ in members], [12_345] * 3)
        tails = [d * 1000 for _, d in members]
        self.assertEqual(tails, sorted(tails))
        self.assertTrue(all(12_345 < end <= request.duration_us for end in tails))
        self.assertTrue(all(end % 1000 == 0 for end in tails))

    def test_reverse_carries_complete_packages_and_encodes_audio_once(self):
        request, inputs = self.copy_example()
        groups = valid_groups(inputs, request)
        clean = {key: value.clone() for key, value in self.clean.items()}
        clean['count'].zero_()
        clean['offset'].zero_()
        clean['tail'].zero_()
        clean['count'][0, 0, 0] = 2
        copy = int((inputs['canvas_head_us'][0, 0] == 12_345).nonzero()[0])
        clean['offset'][0, 0, 0, :2] = copy
        mask = groups.clone()
        mask[0, 0, 0] = False
        initial = corrupt_packages(clean, groups, self.steps, 4, mask=mask)
        with torch.no_grad():
            self.model.count_gate.weight.zero_()
            self.model.count_gate.bias.fill_(-100.)
        with patch.object(self.model, 'encode_audio', wraps=self.model.encode_audio) as encode:
            events, diagnostic = sample_group_diffusion(self.model, RejectTargetReads(inputs),
                request, seed=91, initial_state=initial)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(events, [RhythmEvent(12_345, 12_345, 'tap')] * 2)
        for key in initial:
            torch.testing.assert_close(initial[key][0, 0, 0], diagnostic['state'][key][0, 0, 0], atol=0, rtol=0)
        self.assertEqual(diagnostic['trace'][-1]['remaining_mask'], 0)
        self.assertEqual(len(diagnostic['trace']), 4)

    def test_overflow_preserves_failed_state_without_truncating_to_success(self):
        request = replace(self.request, max_events=1)
        with torch.no_grad():
            self.model.count_gate.weight.zero_()
            self.model.count_gate.bias.fill_(100.)
            self.model.count_positive.weight.zero_()
            self.model.count_positive.bias.fill_(-100.)
            self.model.count_positive.bias.view(4, 16)[:, -1] = 100.
        with self.assertRaisesRegex(GroupSampleError, 'resource overflow') as caught:
            sample_group_diffusion(self.model, self.inputs, request, seed=8)
        error = caught.exception
        self.assertTrue((error.state['count'] == 16).any())
        self.assertGreaterEqual(error.step, 1)

    def test_hold_support_excludes_heads_with_no_native_endpoint(self):
        request = replace(self.request, end_us=1500, duration_us=1500)
        inputs = {k: v.clone() for k, v in self.inputs.items()}
        inputs.update({k: torch.from_numpy(np.asarray(v))[None] for k, v in request_layout(request).items()})
        with torch.no_grad():
            self.model.onset_read.weight.zero_()
            self.model.onset_read.bias.fill_(-10000)
            self.model.onset_read.bias[1] = 10000
            members, _, _ = self.model.ordered_members(torch.randn(200, 32), 0, 1, 1,
                inputs, request, rng=np.random.default_rng(20))
        self.assertEqual(members, [(0, 1)])
        request = replace(request, end_us=500, duration_us=500)
        inputs.update({k: torch.from_numpy(np.asarray(v))[None] for k, v in request_layout(request).items()})
        groups = valid_groups(inputs, request)
        self.assertTrue(groups[0, 0, 0])
        self.assertFalse(groups[0, 0, 1])

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS regression')
    def test_partial_window_and_known_package_masks_match_cpu_mps(self):
        request = replace(self.request, start_us=210000000, end_us=214000000, duration_us=416172833)
        inputs = {k:v.clone() for k,v in self.inputs.items()}
        inputs.update({k:torch.from_numpy(np.asarray(v))[None] for k,v in request_layout(request).items()})
        gpu_inputs = {k:v.to('mps') for k,v in inputs.items()}
        groups = valid_groups(inputs, request)
        torch.testing.assert_close(groups, valid_groups(gpu_inputs, request).cpu(), atol=0, rtol=0)
        events = [RhythmEvent(210004000,212012000,'hold'),RhythmEvent(210005000,210005000,'tap')]
        target = {k:torch.from_numpy(v)[None] for k,v in encode_canvas(events,request).items()}
        clean = clean_packages(target,request)
        gpu_clean = clean_packages({k:v.to('mps') for k,v in target.items()},request)
        for k in clean:torch.testing.assert_close(clean[k],gpu_clean[k].cpu(),atol=0,rtol=0)
        gpu_model = GroupDiffusionCanvas(small_config(),density_reference=4,diffusion_steps=4).to('mps').eval()
        gpu_model.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            cpu_features = self.model.package_features(clean)
            gpu_features = gpu_model.package_features({k:v.to('mps') for k,v in clean.items()}).cpu()
        torch.testing.assert_close(cpu_features,gpu_features,atol=1e-5,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
