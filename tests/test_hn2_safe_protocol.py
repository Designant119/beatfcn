"""Full-contract/fresh-evidence guards plus health and native tensor fixtures."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from tests.test_hn2_admission import StratifiedTests
from tests.test_hn2_runtime import LaunchGateTests
from beatfcn.train.hn2_contract import contract_digest
from beatfcn.train.hn2_safe_runtime import (assess_stratified_health,local_launch_gate,
    REQUIRED_SOURCES,sha,execution_scope,matched_wrong_contexts)


class SafeHealthTests(StratifiedTests):
    def setUp(self):
        super().setUp()
        for population in (self.before,self.after):
            for r in population['rows']:
                for phase in ('p0','p1'):
                    for value in r[phase]['types'].values():value['prior_mean_nll']=2.
        for r in self.free:
            for phase in ('p0','p1'):r[phase]['full50']=.8
        self.free=[deepcopy(r) for seed in range(3) for r in self.free]

    def verdict(self):return assess_stratified_health(self.before,self.after,self.free,[100]*196,self.acceptance)

    def test_loss_gain_cannot_hide_free_interface_failure(self):
        for r in self.free:r['p1']['full50']=.3
        self.assertFalse(self.verdict()['passed'])

    def test_p1_rare_positive_must_beat_initial_and_prior(self):
        self.after['rows'][0]['p1']['types']['drag']['mean_nll']=1.1
        self.assertFalse(self.verdict()['passed'])


class BindingTests(unittest.TestCase):
    def test_complete_contract_and_nonempty_hashed_live_evidence(self):
        fixture=LaunchGateTests();fixture.setUp();self.addCleanup(fixture.tmp.cleanup)
        c=fixture.contract
        for name in REQUIRED_SOURCES:fixture.put(name,'# frozen\n')
        c.update(execution_stage='cuda_health_only',quality_updates_allowed=0,run_id='c22-health',evaluation_panel=[],
            source_sha256={n:sha(fixture.root/n) for n in REQUIRED_SOURCES},frozen_rates=fixture.rates)
        c['cloud_health'].update(fit_draft_indices=[1736,455,3242,1770,1859],
            acceptance=dict(maximum_total_loss=6,maximum_loss_ratio=.3,maximum_field_ratio=.5))
        connection=json.loads((fixture.root/'connection.json').read_text());connection.update(results={},evidence_sha256={})
        for kind in ('apps','rates','billing'):
            name=kind+'-actual.json';fixture.put(name,{'observed':kind,'checked_at_unix':1000})
            connection['results'][kind]=dict(ok=True,checked_at_unix=1000,evidence_path=name)
            connection['evidence_sha256'][name]=sha(fixture.root/name)
        fixture.put('connection.json',connection)
        for name in ('review.json','local.json'):
            r=json.loads((fixture.root/name).read_text());r.update(source_sha256=c['source_sha256'],
                execution_scope=execution_scope(c),contract_sha256=contract_digest(c));fixture.put(name,r)
        fixture.put('contract.json',c)
        with patch.dict('sys.modules',{'modal':None}):
            self.assertEqual(local_launch_gate(fixture.root,'contract.json',now=1000)['contract']['quality_updates_allowed'],0)
        for field,value in [('resources',{**c['resources'],'timeout_s':100}),('evaluation_panel',[0]),('duration_prior_path','different.json')]:
            altered={**c,field:value};fixture.put('contract.json',altered)
            with self.assertRaises(ValueError):local_launch_gate(fixture.root,'contract.json',now=1000)
        fixture.put('contract.json',c);connection['results']={};fixture.put('connection.json',connection)
        with self.assertRaises(ValueError):local_launch_gate(fixture.root,'contract.json',now=1000)

    def test_fixture_contains_fractional_duplicates_complete_carry_and_far_tail(self):
        from beatfcn.train.hn2_native_fixtures import native_fixtures
        import torch
        cases=native_fixtures();first=cases[0]
        self.assertEqual(first['fixture_metadata']['carry_additions'],2)
        target=first['targets'][0,first['target_valid'][0]].tolist()
        self.assertEqual(target.count([1501,4123456,1,-1]),2)
        self.assertIn([1501,1502,1,-1],target)
        state=dict(events=first['targets'],active=first['target_valid'])
        wrong=matched_wrong_contexts(state,first['inputs'])
        for value in wrong.values():
            self.assertTrue(torch.equal(value['active'],state['active']))
            e=value['events'][0,value['active'][0]]
            self.assertTrue(bool((e[:,1]>=e[:,0]).all()))
        self.assertFalse(bool(cases[1]['inputs']['head_valid'].any()))


if __name__=='__main__':unittest.main()
