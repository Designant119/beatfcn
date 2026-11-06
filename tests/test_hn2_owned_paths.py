"""Execute actual worker bodies locally with a Volume symlink and no model."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from beatfcn.train.hn2_owned_paths import checked_owned_paths
from beatfcn.train.hn2_safe_runtime import (
    owned_prefixes, sha, collect_health_evidence, cleanup_owned_files)


ROOT = Path(__file__).resolve().parents[1]


def worker_body(source, namespace):
    tree = ast.parse((ROOT/source).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    fn.decorator_list = []
    code = compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT/source), 'exec')
    exec(code, namespace)
    return namespace['train']


class PathTests(unittest.TestCase):
    def test_plain_and_mount_alias_are_same_owned_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            real = Path(temp).resolve()/'physical';real.mkdir()
            alias = Path(temp)/'data';alias.symlink_to(real, target_is_directory=True)
            assets, output = checked_owned_paths(alias, 'run05', alias/'hn2-assets'/'run05')
            self.assertEqual(assets, real/'hn2-assets'/'run05')
            self.assertEqual(output, real/'runs'/'run05')
            self.assertEqual(checked_owned_paths(real, 'run05', assets), (assets, output))

    def test_wrong_run_and_redirecting_leaf_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            real = Path(temp).resolve();assets = real/'hn2-assets'/'run05';assets.parent.mkdir()
            foreign = real/'foreign';foreign.mkdir();assets.symlink_to(foreign, target_is_directory=True)
            for supplied in (assets, real/'hn2-assets'/'other', Path('relative')):
                with self.assertRaises(ValueError):
                    checked_owned_paths(real, 'run05', supplied)

    def production_fixture(self, repaired):
        with tempfile.TemporaryDirectory() as temp:
            physical = Path(temp).resolve()/'physical';physical.mkdir()
            alias = Path(temp)/'data';alias.symlink_to(physical, target_is_directory=True)
            asset_root = alias/'hn2-assets'/'run05';asset_root.mkdir(parents=True)
            contract = dict(run_id='run05', candidate_id='fixture', source_sha256={}, data_manifest_sha256={})
            contract_json = json.dumps(contract)
            receipt = dict(passed=True, candidate_id='fixture', source_sha256={}, data_manifest_sha256={},
                contract_sha256=hashlib.sha256(contract_json.encode()).hexdigest())
            admission = asset_root/'remote-asset-receipt.json';admission.write_text(json.dumps(receipt))
            untouched = physical/'runs'/'other';untouched.mkdir(parents=True)
            (untouched/'sentinel').write_text('retain')
            commits = []; calls = []
            def run_cuda(c, asset, output, **kwargs):
                calls.append((c, asset, output))
                folder = Path(output)/c['run_id'];folder.mkdir(parents=True)
                (folder/'fixture.json').write_text('{"scope":"filesystem-only;no model"}')
                return dict(status='filesystem_fixture_only', health=None)
            namespace = dict(Path=Path, json=json, hashlib=hashlib, DATA=str(alias), owned_prefixes=owned_prefixes,
                checked_owned_paths=checked_owned_paths, sha=sha, collect_health_evidence=collect_health_evidence,
                cleanup_owned_files=cleanup_owned_files, data=SimpleNamespace(commit=lambda: commits.append(True)),
                envelopes=lambda *args: (contract, None, None, None))
            source = 'beatfcn/cloud/hn2_physical_path_app.py' if repaired else 'beatfcn/cloud/hn2_safe_app.py'
            fn = worker_body(source, namespace)
            kwargs = dict(contract_json=contract_json,authorization_json='{}',review_json='{}',local_gate_json='{}',
                asset_root=str(asset_root),asset_receipt_sha256=sha(admission),dispatch_deadline_unix=1.)
            with patch('beatfcn.train.hn2_safe_runtime.run_cuda', run_cuda):
                if not repaired:
                    with self.assertRaisesRegex(ValueError, 'Private asset prefix'):
                        fn(**kwargs)
                    self.assertEqual(calls, [])
                else:
                    result = fn(**kwargs)
                    self.assertEqual(result['status'], 'filesystem_fixture_only')
                    self.assertEqual(len(calls), 1)
                    self.assertTrue(result['cleanup']['passed'])
                    self.assertIn('fixture.json', result['evidence'])
                    self.assertFalse(asset_root.exists())
                    self.assertFalse((physical/'runs'/'run05').exists())
                    self.assertEqual(len(commits), 1)
                self.assertEqual((untouched/'sentinel').read_text(), 'retain')

    def test_old_actual_worker_body_reproduces_mount_alias_failure(self):
        self.production_fixture(False)

    def test_repaired_actual_worker_body_validates_receipt_handoffs_and_cleans(self):
        self.production_fixture(True)


if __name__ == '__main__':
    unittest.main()
