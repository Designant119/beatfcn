from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import modal
import numpy as np

from beatfcn.cloud.full_database_preparation_app import (
    classify_raw_unverified, completed_report, load_partial_records, plan_raw_verification, unique_osu_rows,
    validate_target_marker, verify_mel_file)
from beatfcn.train.full_database import RULE, digest, prepare_chart, safe_key
from scripts.run_full_database_cloud_preparation import normalized_rates, poll_call


class PreparationRuntimeTests(unittest.TestCase):
    def test_polling_timeout_is_not_a_launcher_failure(self):
        class Call:
            def __init__(self):
                self.calls = 0

            def get(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    # Exact Modal 1.6.2 _functions.poll_function behavior.
                    raise TimeoutError()
                return {'complete': True}

        call = Call()
        result = poll_call(call, modal, threading.Event(), time.time() + 5)
        self.assertEqual(result, {'complete': True})
        self.assertEqual(call.calls, 2)

    def test_modal_polling_exception_is_also_accepted(self):
        class Call:
            def __init__(self):
                self.calls = 0

            def get(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise modal.exception.TimeoutError()
                return {'complete': True}

        call = Call()
        self.assertTrue(poll_call(call, modal, threading.Event(), time.time() + 5)['complete'])
        self.assertEqual(call.calls, 2)

    def test_remote_function_timeout_is_not_swallowed(self):
        class Call:
            def get(self, timeout):
                raise modal.exception.FunctionTimeoutError('remote deadline')

        with self.assertRaises(modal.exception.FunctionTimeoutError):
            poll_call(Call(), modal, threading.Event(), time.time() + 5)

    def test_osu_audio_is_hashed_once_and_extension_conflicts_fail(self):
        rows = [
            dict(source='osu', chart_id='a', audio_sha256='x', audio_ext='ogg'),
            dict(source='osu', chart_id='b', audio_sha256='x', audio_ext='ogg'),
            dict(source='osu', chart_id='c', audio_sha256='y', audio_ext='mp3'),
            dict(source='phira', chart_id='d', audio_sha256='z'),
        ]
        self.assertEqual([row['audio_sha256'] for row in unique_osu_rows(rows)], ['x', 'y'])
        rows[1]['audio_ext'] = 'mp3'
        with self.assertRaisesRegex(ValueError, 'conflicting extensions'):
            unique_osu_rows(rows)

    def test_raw_plan_deduplicates_globally_and_preserves_cross_source_provenance(self):
        rows = [
            dict(source='phira', chart_id='p1', native_id='20', audio_sha256='shared'),
            dict(source='phira', chart_id='p2', native_id='10', audio_sha256='shared'),
            dict(source='phira', chart_id='p3', native_id='30', audio_sha256='phira-only'),
            dict(source='phira', chart_id='p4', native_id='40', audio_sha256='phira-only'),
            dict(source='osu', chart_id='o1', native_id='1', audio_sha256='shared', audio_ext='ogg'),
            dict(source='osu', chart_id='o2', native_id='2', audio_sha256='shared', audio_ext='ogg'),
        ]
        osu, phira, provenance = plan_raw_verification(rows)
        self.assertEqual([row['audio_sha256'] for row in osu], ['shared'])
        self.assertEqual(phira, {'30': [rows[2]]})
        self.assertEqual(len(provenance['shared']), 3)
        self.assertEqual(len(provenance['phira-only']), 2)
        self.assertEqual({row['audio_sha256'] for row in osu} | {
            row['audio_sha256'] for values in phira.values() for row in values},
            {'shared', 'phira-only'})

    def test_raw_content_failure_is_allowed_only_for_pre_quarantined_audio(self):
        rows = [dict(audio_sha256='bad', quarantined=True, kept=False),
                dict(audio_sha256='kept-bad', quarantined=False, kept=True),
                dict(audio_sha256='shared', quarantined=True, kept=False),
                dict(audio_sha256='shared', quarantined=False, kept=True)]
        records = [dict(audio_sha256='bad', source='phira', archive='bad.zip', verified=False),
                   dict(audio_sha256='kept-bad', source='osu', archive='bad.ogg', verified=False),
                   dict(audio_sha256='shared', source='phira', archive='shared.zip', verified=False),
                   dict(audio_sha256='good', source='osu', archive='good.ogg', verified=True)]
        quarantined, unexpected = classify_raw_unverified(records, rows)
        self.assertEqual([value['audio_sha256'] for value in quarantined], ['bad'])
        self.assertEqual(unexpected, ['kept-bad', 'shared'])
        self.assertFalse(quarantined[0]['kept'])

    def test_mel_verifier_binds_frames_and_uses_bytes_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'mel10').mkdir()
            audio = 'a' * 64
            np.save(root / 'mel10' / (audio + '.npy'),
                np.zeros((4, 128), dtype=np.float16), allow_pickle=False)
            (root / 'mel10' / (audio + '.json')).write_text(json.dumps(
                dict(frames=4, hop_sec=.01, dtype='float16')))
            result = verify_mel_file(root, audio)
            self.assertTrue(result['verified'])
            self.assertEqual(result['expected_frames'], 4)
            self.assertGreater(result['bytes'], 0)
            (root / 'mel10' / (audio + '.json')).write_text(json.dumps(
                dict(frames=8, hop_sec=.01, dtype='float16')))
            result = verify_mel_file(root, audio)
            self.assertFalse(result['verified'])
            self.assertEqual(result['expected_frames'], 8)

    def test_target_marker_rejects_incomplete_outcome_accounting(self):
        rows = [dict(chart_id='a'), dict(chart_id='b')]
        marker = dict(batch=0, manifest_sha256='m', planned=['a', 'b'],
            successes=[dict(chart_id='a')], permanent_failures=[],
            success_count=1, permanent_failure_count=0)
        with self.assertRaisesRegex(ValueError, 'outcome accounting'):
            validate_target_marker(marker, 0, rows, 'm')

    def test_existing_target_receipt_rebinds_manifest_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / 'canonical.json'
            row = dict(chart_id='chart', audio_sha256='audio', split='train')
            canonical.write_text(json.dumps(dict(audio_sha256='audio', mode='mania_4k')))
            source_sha = digest(canonical)
            identity = safe_key(row['chart_id'] + ':' + source_sha)
            folder = root / 'targets' / 'targets' / identity
            folder.mkdir(parents=True)
            files = {}
            for name in ('native.npy', 'filtered.npy', 'removed_drag.npy'):
                value = np.zeros((1, 4), dtype=np.int64) if name != 'removed_drag.npy' else np.zeros(1, dtype=bool)
                np.save(folder / name, value, allow_pickle=False)
                files[name] = dict(bytes=(folder / name).stat().st_size,
                    sha256=digest(folder / name))
            receipt = dict(chart_id='chart', audio_sha256='audio', split='train',
                canonical_sha256=source_sha, rule=RULE,
                statistics=dict(model_shift_us=0), files=files,
                target_path=str(folder / 'filtered.npy'))
            (folder / 'receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(prepare_chart(row, canonical, root / 'targets')['chart_id'], 'chart')
            changed = dict(row, split='val')
            with self.assertRaisesRegex(ValueError, 'manifest row'):
                prepare_chart(changed, canonical, root / 'targets')
            (folder / 'filtered.npy').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'target changed'):
                prepare_chart(row, canonical, root / 'targets')

    def test_chunked_receipts_resume_only_verified_unique_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = root / 'partial.jsonl'
            partial.write_text('\n'.join([
                json.dumps(dict(audio_sha256='a', verified=True, manifest_sha256='m')),
                json.dumps(dict(audio_sha256='b', verified=False, manifest_sha256='m')),
                json.dumps(dict(audio_sha256='outside', verified=True, manifest_sha256='m')),
                json.dumps(dict(audio_sha256='c', verified=True, manifest_sha256='other')),
            ]) + '\n')
            records = load_partial_records(partial, {'a', 'b', 'c'}, 'm')
            self.assertEqual(set(records), {'a'})
            partial.write_text(partial.read_text() + json.dumps(
                dict(audio_sha256='a', verified=True, manifest_sha256='m')) + '\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate audio'):
                load_partial_records(partial, {'a', 'b'}, 'm')

    def test_completed_report_requires_matching_final_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / 'content.jsonl';destination.write_text('payload\n')
            report = root / 'receipt.json'
            report.write_text(json.dumps(dict(manifest_sha256='m', complete=False)))
            self.assertIsNone(completed_report(report, destination, 'm'))
            report.write_text(json.dumps(dict(manifest_sha256='m', complete=True, sha256='bad')))
            with self.assertRaisesRegex(ValueError, 'final JSONL'):
                completed_report(report, destination, 'm')
            report.write_text(json.dumps(dict(manifest_sha256='m', complete=True,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
            self.assertTrue(completed_report(report, destination, 'm')['complete'])

    def test_cpu_rate_is_not_halved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'reports').mkdir()
            (root / 'reports/rhythm_v2_rates.json').write_text(json.dumps(
                dict(cpu_hour_cost='0.04730', mem_gib_hour_cost='0.00800')))
            with mock.patch('scripts.run_full_database_cloud_preparation.ROOT', root):
                rates = normalized_rates()
            self.assertEqual(Decimal(rates['cpu_usd_s']), Decimal('0.04730') / Decimal(3600))
            self.assertEqual(Decimal(rates['memory_gib_usd_s']), Decimal('0.00800') / Decimal(3600))


if __name__ == '__main__':
    unittest.main()
