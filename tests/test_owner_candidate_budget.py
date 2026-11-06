"""Paid dispatch must fail closed without spending older campaign/probe funds."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from beatfcn.train.owner_candidate_budget import (
    authorize_owner_candidate, candidate_gate, digest,
    record_owner_candidate_usage, reserve_owner_candidate,
)


GRANT='owner-modal120m-20261030'


class OwnerCandidateBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name,text in [('source.py','source'),('data.json','{}'),('review.md','independent review'),('owner.txt','explicit cap')]:
            (self.root/name).write_text(text)
        self.contract=dict(candidate_id='candidate',grant_id=GRANT,
            actual_trainable_parameters=138621056,
            independent_review_receipt='review.json',local_gate_receipt='local.json',
            source_sha256={'source.py':digest(self.root/'source.py')},
            data_manifest_sha256={'data.json':digest(self.root/'data.json')},
            density_rule='all-events-1s-gt12-drag-only-v2',evaluate_every=250)
        for name in ['review.json','local.json']:
            receipt=dict(passed=True,candidate_id='candidate',
                source_sha256=self.contract['source_sha256'],
                data_manifest_sha256=self.contract['data_manifest_sha256'],unresolved_blockers=[],
                independent_original_reports=[dict(path='review.md',sha256=digest(self.root/'review.md'))])
            (self.root/name).write_text(json.dumps(receipt))
        self.old=dict(approved_total_usd=80,planned_stages={'old':{'reserved_upper_usd':73.8}},
                      cloud_probe_pool={'cap':6,'used':0})
        self.ledger=self.root/'ledger.json'
        self.ledger.write_text(json.dumps(dict(**self.old,conditional_candidate_authorizations={
            GRANT:dict(owner_authorized=False,new_usd_cap=None)})))
        self.resources=dict(timeout_s=100,startup_allowance_s=10,stop_allowance_s=10,
            cpu=2,memory_gib=4,max_containers=1,retries=0,other_upper_usd=.1)
        self.rates=dict(gpu_usd_s=.01,cpu_usd_s=.001,memory_gib_usd_s=.0001)
        self.connection=dict(fresh_check_succeeded=True,checked_at_unix=time.time(),
            evidence_sha256='evidence',full_lifecycle_bound_verified=True)

    def authorize(self,cap=2):
        authorize_owner_candidate(self.ledger,GRANT,cap,owner_evidence_path='owner.txt',root=self.root)

    def reserve(self,**kwargs):
        return reserve_owner_candidate(self.ledger,'job',self.contract,self.root,
            kwargs.get('resources',self.resources),self.rates,kwargs.get('connection',self.connection))

    def test_pending_cap_and_insufficient_cap_never_mutate_old_funds(self):
        before=self.ledger.read_text()
        with self.assertRaisesRegex(ValueError,'cap pending'):self.reserve()
        self.assertEqual(before,self.ledger.read_text())
        self.authorize(.5)
        with self.assertRaisesRegex(ValueError,'cap exhausted'):self.reserve()
        after=json.loads(self.ledger.read_text())
        self.assertEqual(self.old,{k:after[k] for k in self.old})
        self.assertEqual([],after['conditional_candidate_authorizations'][GRANT]['jobs'])

    def test_source_data_review_and_local_receipt_must_match(self):
        self.assertEqual([],candidate_gate(self.contract,self.root))
        for name in ['source.py','data.json','review.md']:
            original=(self.root/name).read_text();(self.root/name).write_text('drift')
            self.assertTrue(candidate_gate(self.contract,self.root),name)
            (self.root/name).write_text(original)
        receipt=json.loads((self.root/'local.json').read_text());receipt['source_sha256']={}
        (self.root/'local.json').write_text(json.dumps(receipt))
        self.assertTrue(candidate_gate(self.contract,self.root))

    def test_reservation_uses_hard_limits_and_no_duplicate_retry(self):
        self.authorize();job=self.reserve()
        self.assertAlmostEqual(float(job['reserved_upper_usd']),1.588)
        with self.assertRaises(ValueError):self.reserve()
        bad=copy.deepcopy(self.resources);bad['cpu']=0
        with self.assertRaisesRegex(ValueError,'Positive hard'):self.reserve(resources=bad)
        bad=copy.deepcopy(self.connection);bad['checked_at_unix']=time.time()-601
        with self.assertRaisesRegex(ValueError,'Fresh'):self.reserve(connection=bad)

    def test_live_meter_monotonic_and_requires_actual_stop_evidence(self):
        self.authorize();self.reserve()
        def update(elapsed,**kw):
            return record_owner_candidate_usage(self.ledger,'job',elapsed_s_since_dispatch=elapsed,app_id='ap-real',**kw)
        first=update(20,reported_actual_usd=.2)
        self.assertFalse(first['job']['stop_required'])
        with self.assertRaisesRegex(ValueError,'backwards'):update(19)
        with self.assertRaisesRegex(ValueError,'backwards'):update(21,reported_actual_usd=.1)
        with self.assertRaisesRegex(ValueError,'app-stop'):update(21,closed=True)
        near=update(111)
        self.assertTrue(near['job']['stop_required'])
        self.assertEqual('stopping',near['job']['status'])
        update(112,closed=True,stop_evidence={'app_state':'APP_STATE_STOPPED','receipt':'actual.json'})
        with self.assertRaisesRegex(ValueError,'cannot resume'):update(113)
        after=json.loads(self.ledger.read_text())
        self.assertEqual(self.old,{k:after[k] for k in self.old})


if __name__=='__main__':unittest.main()
