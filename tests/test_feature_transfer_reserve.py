import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

from scripts.fetch_frozen_volume_files import check_storage, transfer_prior


class FeatureTransferReserveTests(unittest.TestCase):
    def test_remaining_payload_must_fit_above_reserve(self):
        with patch('scripts.fetch_frozen_volume_files.shutil.disk_usage', return_value=SimpleNamespace(free=120)):
            check_storage({'minimum_free_bytes': 100}, 20)
            with self.assertRaisesRegex(ValueError, 'reserve'):
                check_storage({'minimum_free_bytes': 100}, 21)

    def test_legacy_plan_does_not_acquire_a_new_constraint(self):
        with patch('scripts.fetch_frozen_volume_files.shutil.disk_usage', side_effect=AssertionError):
            check_storage({}, 20)

    def test_policy_revalidation_resume_keeps_spent_bytes_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = dict(local_root='cache', files=[dict(path='file.npy', bytes=40)],
                          maximum_transfer_bytes=100, minimum_free_bytes=200)
            p = root/'plan.json'; p.write_text(json.dumps(parent))
            phash = hashlib.sha256(p.read_bytes()).hexdigest()
            prior = dict(plan_sha256=phash, cumulative_bytes=45, ended_at_unix=1,
                         complete=False, files=[dict(path='file.npy', sha256='stored')])
            r = root/'receipt.json'; r.write_text(json.dumps(prior))
            plan = dict(parent, resume_from=dict(plan='plan.json', plan_sha256=phash,
                        receipt='receipt.json', receipt_sha256=hashlib.sha256(r.read_bytes()).hexdigest()))
            with patch('scripts.fetch_frozen_volume_files.ROOT', root):
                self.assertEqual(transfer_prior(plan, 'new', root/'new.json'), prior)
                with self.assertRaisesRegex(ValueError, 'maximum_transfer_bytes'):
                    transfer_prior(dict(plan, maximum_transfer_bytes=101), 'new', root/'new.json')
                with self.assertRaisesRegex(ValueError, 'files'):
                    transfer_prior(dict(plan, files=[]), 'new', root/'new.json')
                r.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'evidence changed'):
                    transfer_prior(plan, 'new', root/'new.json')
