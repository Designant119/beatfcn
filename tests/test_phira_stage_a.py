import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch

from beatfcn.events.canonical import Event, LineLoc, ModelView
from beatfcn.events.phira import parse_phira
from beatfcn.events.requests import build_reference, build_request
from beatfcn.events.reference_policy import bound_evidence, content_fingerprint, controlled_pair
from beatfcn.model import features as F
from beatfcn.model.batch import collate
from beatfcn.model.evaluation import generate_song, score_events, evaluate_benchmark
from beatfcn.model.space import SpaceEncoder
from beatfcn.model.drag_benchmark import score_drag_segments
from beatfcn.train.data import chart_from_canon
from beatfcn.train.stage_a import (ARMS, build_triplet, prepare_sample, protocol, warm_start, tensor_fingerprint,
                                 GeometryPreparer, prepare_core_schedule, FrozenArmSampler)
from beatfcn.spatial.rpe import digest
from beatfcn.spatial import compile_rpe, project
from beatfcn.spatial.context import Packet, STATUS
from events_helpers import DURATION
from test_p1_model import CFG, song

ROOT = Path(__file__).resolve().parents[1]


def document():
    return {'META': {'offset': 0, 'RPEVersion': 170}, 'BPMList': [{'startTime': [0, 0, 1], 'bpm': 120}],
            'judgeLineList': [{'father': -1, 'bpmfactor': 1, 'eventLayers': [{
                'moveXEvents': [{'startTime': [0, 0, 1], 'endTime': [60, 0, 1], 'start': 0, 'end': 150, 'easingType': 1}],
                'moveYEvents': [], 'rotateEvents': []}], 'notes': [
                {'type': 1 if i % 3 else 4, 'startTime': [i, 0, 1], 'endTime': [i, 0, 1],
                 'positionX': 270 if i % 2 else -270, 'above': 1, 'isFake': 0} for i in range(1, 55)]}]}


def fixture(kind='middle_holes', seed=4, doc=None):
    raw = json.dumps(document() if doc is None else doc).encode()
    chart, scene = parse_phira(raw), compile_rpe(raw)
    sample = build_request(chart, kind=kind, seed=seed, audio_id='synthetic', audio_duration_us=DURATION,
                           control_dropout=False)
    prepared, context = prepare_sample(sample, scene,
                                       reference_sources={r['reference_id']: scene for r in sample.evidence})
    return chart, scene, prepared, context


class DataContractTests(unittest.TestCase):
    def test_submillisecond_permission_start_does_not_allow_previous_grid_point(self):
        chart, _, sample, context = fixture('from_scratch')
        sample.request['regions'][0]['start_us'] = 501
        # Reproject because permissions changed; do not reuse a cached context.
        context = project(None, sample.request, sample.evidence)
        ex = F.build_core(sample, chart.events, 0, DURATION, stage_a_arm='removed', spatial_context=context)
        self.assertFalse(ex.bin_ok[0, 0])
        self.assertTrue(ex.bin_ok[0, 1])

    def test_reference_geometry_never_exceeds_public_span(self):
        _, scene, sample, _ = fixture('reference')
        sample.request['references'][0]['span']['end_us'] -= 1
        with self.assertRaisesRegex(ValueError, 'span or coverage'):
            project(scene, sample.request, sample.evidence)

    def test_fake_and_ambiguous_canon_are_rejected_not_silently_dropped(self):
        event = Event('a', 1, 1, 'drag', LineLoc('0', 'above', 0.0)).to_json()
        doc = {'events': [event], 'mode': 'phira', 'source_format': 'phira_rpe'}
        self.assertEqual(len(chart_from_canon(doc).events), 1)
        for value in (True, 1, 'false', None, 0.0):
            for field in ('fake', 'isFake'):
                changed = copy.deepcopy(doc)
                changed['events'][0][field] = value
                with self.assertRaises(ValueError):
                    chart_from_canon(changed)

    def test_reference_budget_preserves_all_positives_and_complete_spans(self):
        events = [Event(str(i), i * 100_000, i * 100_000, 'drag', LineLoc('0', 'above', 0.0)) for i in range(160)]
        evidence = [build_reference(events, (0, 8_000_000), 'complete', np.random.default_rng(0)),
                    build_reference(events, (8_000_000, 16_000_000), 'complete', np.random.default_rng(1))]
        before = copy.deepcopy(evidence)
        bounded, audit = bound_evidence(evidence)
        self.assertEqual(sum(len(r['selected']) + len(r['unselected_us']) for r in bounded), 256)
        self.assertEqual(audit['negative_rows_removed'], 64)
        for a, b in zip(before, bounded):
            self.assertEqual(a['selected'], b['selected'])
            self.assertEqual(a['span'], b['span'])
        self.assertEqual(evidence, before)
        with self.assertRaises(ValueError):
            bound_evidence(evidence, max_rows=100)

    def test_reference_span_shortening_does_not_split_chord(self):
        events = [Event(str(i), 1_000_000, 1_000_000, 'drag', LineLoc('0', 'above', float(i))) for i in range(129)]
        ref = build_reference(events, (0, 2_000_000), 'complete', np.random.default_rng(0))
        self.assertEqual(ref['span'], (0, 1_000_000))
        self.assertEqual(ref['selected'], [])

    def test_identical_reference_ids_do_not_manufacture_style_contrast(self):
        chart, _, sample, _ = fixture('reference')
        a, b = copy.deepcopy(sample.evidence), copy.deepcopy(sample.evidence)
        for ref in b:
            for e in ref['selected']:
                e['event_id'] += ':renamed'
            ref['unselected_us'] = []
            ref['unselected_source'] = []
        self.assertEqual(content_fingerprint(a), content_fingerprint(b))
        with self.assertRaises(ValueError):
            controlled_pair(a, b)
        b[0]['selected'][0]['type'] = 'flick'
        aa, bb = controlled_pair(a, b)
        self.assertTrue(all(not r['unselected_us'] for r in aa + bb))

    def test_empty_target_success_is_not_zero_quality_or_free_one(self):
        score = score_events([], [], 'phira')
        self.assertTrue(score['empty_target_silent'])
        self.assertIsNone(score['note_20ms']['f1'])
        self.assertIsNone(score['onset_20ms']['f1'])
        note = Event('a', 1, 1, 'drag', LineLoc('0', 'above', 0.0))
        self.assertEqual(score_events([note], [], 'phira')['note_20ms']['f1'], 0)

    def test_real_previously_failing_capacity_requests(self):
        manifest = json.loads((ROOT / 'bench/capability-v1/manifest.json').read_text())
        for cid in (12870, 56053):
            path = ROOT / f'reports/phira_{cid}_canon_audit.json'
            if not path.exists():
                self.skipTest('local audited canon not present')
            doc = json.loads(path.read_text())
            row = next(r for r in manifest['rows'] if r['chart_id'] == f'phira:{cid}')
            chart = chart_from_canon(doc, shift_us=-2000)
            sample = build_request(chart, kind='reference', seed=row['benchmark_seed'], audio_id=row['audio_sha256'],
                                   audio_duration_us=round(row['audio_sec'] * 1e6), control_dropout=False)
            sample, context = prepare_sample(sample)
            positives = sum(len(r['selected']) for r in sample.evidence)
            for c0 in range(0, sample.request['audio_duration_us'], F.CORE_US):
                triplet = build_triplet(sample, chart.events, c0, context)
                for ex in triplet.values():
                    self.assertEqual(int((ex.refs[:, 7] == 1).sum()), positives)
                    self.assertLessEqual(len(ex.refs), F.MAX_REF)


class SpatialProjectionTests(unittest.TestCase):
    def test_hidden_geometry_and_unobserved_lines_do_not_change_inputs(self):
        chart, scene, sample, context = fixture()
        doc = document()
        hidden_id = sample.hidden[-1].event_id
        li, ni = map(int, hidden_id.split(':')[1:])
        doc['judgeLineList'][li]['notes'][ni]['positionX'] = 99999
        doc['judgeLineList'].append({'eventLayers': [], 'notes': []})
        changed = compile_rpe(json.dumps(doc).encode())
        other = project(changed, sample.request, sample.evidence)
        # Cache identity changes with source bytes, but model-visible packets do not.
        self.assertNotEqual(other.cache_key, context.cache_key)
        self.assertEqual([p.key() for p in other.memory], [p.key() for p in context.memory])
        for key in context.known:
            self.assertEqual(context.known[key][1].key(), other.known[key][1].key())

    def test_generated_prefix_never_recovers_original_geometry_by_id(self):
        chart, _, sample, context = fixture('from_scratch')
        ex = build_triplet(sample, chart.events, 8_000_000, context)['operation']
        generated = ex.known[:, 7] == 1
        self.assertTrue(generated.any())
        self.assertFalse(ex.space['known']['valid'][generated].any())
        self.assertTrue((ex.known[generated, 3:5] == -1).all())
        self.assertTrue((ex.space['known']['status'][generated] == STATUS['unplaced']).all())

    def test_stale_projection_and_opaque_scene_ref_are_rejected(self):
        chart, _, sample, context = fixture()
        req = copy.deepcopy(sample.request)
        req['observed_events'] = []
        with self.assertRaisesRegex(ValueError, 'stale'):
            build_triplet(replace(sample, request=req), chart.events, 0, context)
        _, _, scratch, empty_context = fixture('from_scratch')
        self.assertTrue(scratch.request['scene_ref'])
        self.assertEqual(len(empty_context.memory), 0)

    def test_explicit_retained_scene_grant_is_separate_from_no_scene(self):
        chart, scene, sample, context = fixture('from_scratch')
        self.assertFalse(context.memory)
        grant = {'kind': 'retained_line_scene', 'source_sha256': scene.source_sha256,
                 'line_ids': ['0'], 'span_us': [0, DURATION]}
        prepared, retained = prepare_sample(sample, scene, grants=[grant])
        self.assertEqual(len(retained.memory), 1)
        self.assertEqual(retained.memory[0].source_kind, 3)
        self.assertEqual(sample.hidden, prepared.hidden)
        bad = {**grant, 'source_sha256': 'different-source'}
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            prepare_sample(sample, scene, grants=[bad])

    def test_reference_geometry_is_not_target_binding_permission(self):
        _, scene, sample, context = fixture('reference')
        self.assertTrue(any(p.valid.any() for _, p in context.references.values()))
        self.assertTrue(all(p.source_kind != 2 for p in context.memory))
        req = copy.deepcopy(sample.request)
        for ref in req['references']:
            ref['space_attributes'] = False
        changed = project(scene, req, sample.evidence)
        self.assertTrue(all(not p.valid.any() for _, p in changed.references.values()))

    def test_memory_overflow_is_explicit_and_does_not_drop_events(self):
        _, scene, sample, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'memory'):
            project(scene, sample.request, sample.evidence, max_memory=0)
        ctx = project(scene, sample.request, sample.evidence, max_memory=0, memory_overflow='truncate_reported')
        self.assertGreater(ctx.audit['memory_truncated'], 0)
        self.assertEqual(len(ctx.known), len(sample.request['observed_events']))

    def test_clock_shift_and_discontinuity_are_evaluated_at_actual_time(self):
        doc = document()
        doc['judgeLineList'][0]['eventLayers'][0]['moveXEvents'] = [
            {'startTime': [0, 0, 1], 'endTime': [2, 0, 1], 'start': 0, 'end': 0},
            {'startTime': [2, 0, 1], 'endTime': [60, 0, 1], 'start': 675, 'end': 675}]
        scene = compile_rpe(json.dumps(doc).encode(), info_offset_s=.25)
        self.assertAlmostEqual(scene.region('0', 0, 1_249_999)[2], 0)
        self.assertAlmostEqual(scene.region('0', 0, 1_250_000)[2], 1)
        chart = parse_phira(json.dumps(doc).encode(), info_offset_sec=.25)
        shifted = replace(chart, events=tuple(replace(e, onset_us=e.onset_us-2000, end_us=e.end_us-2000) for e in chart.events))
        sample = build_request(shifted, kind='middle_holes', seed=4, audio_id='a', audio_duration_us=DURATION)
        _, ctx = prepare_sample(sample, scene, shift_us=-2000)
        self.assertGreater(ctx.audit['known_resolved_rows'], 0)

    def test_geometry_matches_checked_real_case_snapshots(self):
        path = ROOT / 'reports/phira_spatial_2026-10-24/cases.json'
        if not path.exists():
            self.skipTest('real-case audit unavailable')
        for case in json.loads(path.read_text()):
            cid = case['chart']
            base = ROOT / 'inputs/phira_spatial_20261024'
            info = json.loads((base / f'{cid}.info.json').read_text())
            scene = compile_rpe((base / f'{cid}.chart.json').read_bytes(), info_offset_s=info.get('offset', 0))
            events = sum(case['sequences'], []) if 'sequences' in case else case.get('events', [case.get('hold')])
            for event in events:
                if event is None:
                    continue
                actual = scene.region(str(event['line']), event['x'], event['audio'] * 1e6,
                                      judge_area=event.get('judge_area', 1))
                np.testing.assert_allclose(actual, [*event['normal'], event['d'], event['half_width']], atol=1e-5)


class StageAModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_three_arms_match_labels_and_new_encoder_has_gradient_without_updates(self):
        chart, _, sample, context = fixture()
        c0 = F.choose_core(sample, np.random.default_rng(0), DURATION)
        exs = build_triplet(sample, chart.events, c0, context)
        from beatfcn.model.p1 import P1Model
        base = P1Model(CFG)
        checkpoint = {'config': asdict(CFG), 'model': base.state_dict()}
        for arm in ARMS:
            model = warm_start(checkpoint, arm)
            bt = collate([exs[arm]], [song(1)], CFG.taps, CFG.top_tap)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            loss, _ = model(bt)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            if arm == 'operation':
                total = sum(float(p.grad.abs().sum()) for p in model.space_encoder.parameters() if p.grad is not None)
                self.assertGreater(total, 0)
            for k, v in before.items():
                self.assertTrue(torch.equal(v, model.state_dict()[k]))

    def test_operation_encoder_uses_geometry_and_distinguishes_unknown(self):
        encoder = SpaceEncoder(32).eval()
        points = torch.zeros(1, 1, 8, 7)
        points[..., 1], points[..., 4] = 1, .2
        valid = torch.ones(1, 1, 8, dtype=torch.bool)
        status, source = torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.long)
        a = encoder(points, valid, status, source)
        b = points.clone(); b[..., 3] = .6
        self.assertGreater(float((a - encoder(b, valid, status, source)).abs().max()), 1e-5)
        flipped = points.clone(); flipped[..., 1:4] *= -1
        torch.testing.assert_close(a, encoder(flipped, valid, status, source))
        self.assertGreater(float((a - encoder(points, valid & False, status * 0, source * 0)).abs().max()), 1e-5)

    def test_line_id_permutation_cannot_change_operation_batch(self):
        doc = document()
        first, second = copy.deepcopy(doc['judgeLineList'][0]), copy.deepcopy(doc['judgeLineList'][0])
        first['notes'], second['notes'] = first['notes'][::2], second['notes'][1::2]
        second['eventLayers'][0]['moveXEvents'][0]['start'] = -125
        doc['judgeLineList'] = [first, second]
        other = copy.deepcopy(doc); other['judgeLineList'].reverse()
        batches = []
        for source in (doc, other):
            chart, _, sample, context = fixture(doc=source)
            c0 = F.choose_core(sample, np.random.default_rng(0), DURATION)
            ex = build_triplet(sample, chart.events, c0, context)['operation']
            batches.append(collate([ex], [song(0)], CFG.taps, CFG.top_tap))
        self.assertEqual(tensor_fingerprint(batches[0]), tensor_fingerprint(batches[1]))

    def test_incremental_encoding_matches_full_with_spatial_memory(self):
        chart, _, sample, context = fixture()
        c0 = F.choose_core(sample, np.random.default_rng(0), DURATION)
        ex = build_triplet(sample, chart.events, c0, context)['operation']
        self.assertGreater(len(ex.groups), 0)
        from beatfcn.model.p1 import P1Model
        model = P1Model(replace(CFG, phira_context='operation')).eval()
        empty = replace(ex, groups=[], right=ex.right[:1])
        one = replace(ex, groups=ex.groups[:1], right=ex.right[:2])
        cache = {}
        with torch.no_grad():
            model.encode(collate([empty], [song(0)], CFG.taps, CFG.top_tap), cache)
            full = collate([one], [song(0)], CFG.taps, CFG.top_tap)
            incremental = model.encode_step(full, cache)[0]
            reference = model.encode(full)[0][:, 1:2]
        torch.testing.assert_close(incremental, reference, atol=2e-5, rtol=2e-5)

    def test_masked_reference_cannot_recover_spatial_attributes(self):
        chart, _, sample, context = fixture('reference')
        c0 = F.choose_core(sample, np.random.default_rng(0), DURATION)
        ex = build_triplet(sample, chart.events, c0, context)['operation']
        bt = collate([ex], [song(0)], CFG.taps, CFG.top_tap)
        bt['refs'][..., 1] = 6  # CF selected with masked attributes
        other = copy.deepcopy(bt)
        other['space_refs_points'] += torch.randn_like(other['space_refs_points']) * 10
        from beatfcn.model.p1 import P1Model
        model = P1Model(replace(CFG, phira_context='operation')).eval()
        with torch.no_grad():
            a, b = model.encode(bt)[0], model.encode(other)[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_actual_model_responds_to_geometry_and_checks_arm(self):
        chart, _, sample, context = fixture()
        c0 = F.choose_core(sample, np.random.default_rng(0), DURATION)
        ex = build_triplet(sample, chart.events, c0, context)['operation']
        bt = collate([ex], [song(0)], CFG.taps, CFG.top_tap)
        other = copy.deepcopy(bt)
        other['space_known_points'][..., 3] += other['space_known_valid'] * .75
        other['space_memory_points'][..., 3] += other['space_memory_point_valid'] * .75
        from beatfcn.model.p1 import P1Model
        model = P1Model(replace(CFG, phira_context='operation')).eval()
        with torch.no_grad():
            self.assertGreater(float((model.encode(bt)[0] - model.encode(other)[0]).abs().max()), 1e-5)
        with self.assertRaisesRegex(ValueError, 'arm'):
            P1Model(CFG).encode(bt)

    def test_whole_song_uses_stage_a_and_generated_unknown_across_cores(self):
        chart, _, sample, context = fixture('from_scratch')
        from beatfcn.model.p1 import P1Model
        model = P1Model(replace(CFG, phira_context='operation')).eval()
        with patch('beatfcn.model.sample._pick', return_value=0):
            result = generate_song(model, sample, song(0), spatial_context=context, greedy=True)
        self.assertTrue(result['complete'], result['failure'])
        self.assertEqual(result['events'], [])
        self.assertFalse(result['placement_checked'])

    def test_frozen_benchmark_replays_all_three_arms_with_one_source_adapter(self):
        chart, _, _, _ = fixture()
        rows = [{'chart_id': 'phira:fixture', 'native_id': 'fixture', 'audio_sha256': 'synthetic',
                 'mode': 'phira', 'split': 'val', 'audio_sec': DURATION / 1e6,
                 'benchmark_seed': 7, 'benchmark_stratum': 'fixture'}]
        kinds = ['middle_holes', 'reference']
        manifest = {'rows': rows, 'rows_sha256': digest(rows), 'seed': 7, 'request_kinds': kinds,
                    'stage_a_arms': list(ARMS), 'requests': [
                        {'chart_id': 'phira:fixture', 'song': 'synthetic', 'mode': 'phira',
                         'kind': k, 'seed': 7, 'stratum': 'fixture'} for k in kinds]}
        manifest['manifest_sha256'] = digest(manifest)
        preparer = GeometryPreparer(lambda row: {'raw': json.dumps(document()).encode(), 'info_offset_s': 0})
        cache = song(0)
        corpus = SimpleNamespace(train_rows=lambda r: None, chart=lambda key: chart,
                                 song=lambda key, in_memory: cache)
        from beatfcn.model.p1 import P1Model
        base = P1Model(CFG)
        checkpoint = {'config': asdict(CFG), 'model': base.state_dict()}
        requests = []
        for arm in ARMS:
            model = warm_start(checkpoint, arm).eval()
            with patch('beatfcn.model.sample._pick', return_value=0):
                result = evaluate_benchmark(model, corpus, manifest, spatial_prepare=preparer)
            self.assertTrue(result['valid_for_comparison'])
            self.assertTrue(all(r['complete'] and r['stage_a_arm'] == arm for r in result['songs']))
            requests.append([r['request'] for r in result['songs']])
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[1], requests[2])

    def test_common_schedule_is_replayable_without_arm_specific_resampling(self):
        chart, _, _, _ = fixture()
        row = {'chart_id': 'phira:fixture', 'audio_sha256': 'synthetic', 'mode': 'phira',
               'audio_sec': DURATION / 1e6}
        preparer = GeometryPreparer(lambda row: {'raw': json.dumps(document()).encode(), 'info_offset_s': 0})
        specs = [{'row': row, 'seed': 4, 'kind': 'middle_holes', 'core_start_us': 0},
                 {'row': row, 'seed': 7, 'kind': 'reference', 'core_start_us': 8_000_000}]
        schedule, audit = prepare_core_schedule(specs, lambda key: chart, preparer)
        again, second = prepare_core_schedule(specs, lambda key: chart, preparer)
        self.assertEqual(audit['accepted'], 2, audit)
        self.assertEqual(audit['schedule_sha256'], second['schedule_sha256'])
        for arm in ARMS:
            sampler = FrozenArmSampler(schedule, arm, lambda key: key)
            examples = [e for e, cache in sampler.draw(2)]
            self.assertEqual([e.provenance['rhythm_targets_sha256'] for e in examples],
                             [r['rhythm_targets_sha256'] for r in audit['schedule']])
            self.assertEqual(sampler.skipped, {})
            with self.assertRaisesRegex(ValueError, 'exhausted'):
                sampler.draw(1)

    def test_drag_diagnostics_do_not_claim_ioi_is_gesture_ground_truth(self):
        notes = [Event(str(i), i * 100_000, i * 100_000, 'drag', LineLoc('0', 'above', 0.0)) for i in range(5)]
        case = {'case_id': 'candidate', 'span_us': [0, 600_000], 'semantic_status': 'temporal_candidate'}
        row = score_drag_segments(notes * 2, notes, [case])['cases'][0]
        self.assertEqual(row['count_ratio'], 2)
        self.assertEqual(row['true'], 5)
        self.assertEqual(row['operation_quality'], 'not_measured_without_actions')
        self.assertFalse(protocol()['launch_enabled'])


if __name__ == '__main__':
    unittest.main()
