import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from beatfcn.train.campaign_budget import reserve
from beatfcn.train.cloud_probe_budget import (
    POOL, authorize_probe_pool, cloud_probe_gate, scaling_gate, reserve_probe, record_probe_usage,
)


ROOT=Path(__file__).resolve().parents[1]


class CloudProbeBudgetTests(unittest.TestCase):
    def setUp(self):
        registry = [ROOT/'reports/local-portfolio-v1/mechanism-reviews-v2/index.json',
                    ROOT/'reports/local-evolution-v1/mechanism-reviews/index.json']
        if not any(path.exists() for path in registry):
            raise unittest.SkipTest('mechanism-review registry artifacts are not in this checkout')
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.path=Path(tmp.name)/'ledger.json'
        self.local=Path(tmp.name)/'local-gate.json';self.local.write_text('{"passed":true}')
        initial=dict(approved_total_usd=80.,historical_accounted_upper_usd=3.,planned_other_reserve_usd=1.,
            planned_stages={'old':dict(reserved_upper_usd=60.)},paid_stages_dispatched=[dict(name='paid',reserved_usd=9.,status='failed')])
        self.path.write_text(json.dumps(initial));self.initial=initial
        self.resources=dict(cpu=4,memory_gib=16,gpu='H100',timeout_s=2400,startup_allowance_s=120,
            stop_allowance_s=30,max_containers=1,retries=0)
        self.rates=dict(cpu_hour_cost='.01',mem_gib_hour_cost='.001',gpu_hour_cost_h100='1')
        self.connection=dict(fresh_check_succeeded=True,checked_at_unix=time.time(),
            full_lifecycle_bound_verified=True,evidence_sha256='fixture-rates-and-billing')
        source='beatfcn/model/local_evolution_controls.py';digest=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        self.contract=dict(id='probe-fixture',experiment_class='cloud_probe',purpose='causal_control',
            single_intervention='Erase cross-chunk state only',predecessors=['ssd-reference1000'],
            shared_failure_modes=[],mechanism_changes={},execution_ready=True,
            pretraining_probe_receipts=[dict(status='completed',scope='new_implementation',evidence='reports/local-evolution-v1/preflight.json')],
            local_gate_receipt=str(self.local),baseline_identity={source:digest},source_sha256={source:digest})

    def reserve(self,name='p1',resources=None):
        return reserve_probe(self.path,name,self.contract,ROOT,resources or self.resources,self.rates,
                             self.connection,other_upper_usd='.02')

    def test_authorization_is_idempotent_and_preserves_old_reservations(self):
        authorize_probe_pool(self.path);authorize_probe_pool(self.path);r=json.loads(self.path.read_text())
        self.assertEqual(r['approved_total_usd'],80.)
        self.assertEqual(r['paid_stages_dispatched'],self.initial['paid_stages_dispatched'])
        self.assertEqual(r['planned_stages']['old'],self.initial['planned_stages']['old'])
        self.assertEqual(r['campaign_committed_upper_usd'],79.)
        self.assertEqual(len(r['cloud_probe_events']),1)

    def test_small_probe_passes_without_scaling_quality_gates(self):
        self.assertEqual(cloud_probe_gate(self.contract,ROOT),[])
        self.assertTrue(scaling_gate(self.contract,ROOT,scaling_authorized=True))

    def test_scaling_cannot_use_small_probe_entry(self):
        plan=copy.deepcopy(self.contract);plan['scaling_run']=True
        self.assertIn('Scaling cannot spend the probe grant',cloud_probe_gate(plan,ROOT))

    def test_scaling_has_its_own_receipt_and_still_needs_its_own_authorization(self):
        receipt=self.local.parent/'scaling.json'
        gates={k:True for k in ('purpose_isolated','matched_local_evidence','new_components_frozen',
                               'numerical_resources_checked','acceptance_registered')}
        receipt.write_text(json.dumps(dict(experiment_class='scaling',passed=True,gates=gates)))
        plan=dict(experiment_class='scaling',primary_claim='capacity',matched_control='fixture',
            frozen_new_components=['fixture-component'],numerical_resource_receipt=str(self.local),
            scaling_entry_receipt=str(receipt),required_reports=['per_type','empty_windows','reference_masked','hold_tails'],
            preregistered_acceptance=dict(primary_metric='onset10',baseline_value=.3,minimum_gain=.02,
                maximum_regressions={'full_note_f1':.01},fixed_endpoint=1000))
        self.assertEqual(scaling_gate(plan,ROOT,scaling_authorized=True),[])
        self.assertIn('Scaling authorization absent; probe grant is not scaling authorization',scaling_gate(plan,ROOT))
        del plan['preregistered_acceptance']['maximum_regressions']
        self.assertTrue(scaling_gate(plan,ROOT,scaling_authorized=True))

    def test_missing_local_gate_or_changed_source_blocks_dispatch(self):
        self.local.write_text('{"passed":false}')
        self.assertIn('Local gate has not passed',cloud_probe_gate(self.contract,ROOT))
        self.contract['source_sha256'][next(iter(self.contract['source_sha256']))]='0'*64
        self.assertTrue(any('Frozen identity' in v for v in cloud_probe_gate(self.contract,ROOT)))

    def test_old_dispatch_entry_cannot_spend_new_scope(self):
        authorize_probe_pool(self.path)
        with self.assertRaisesRegex(ValueError,'small-probe pool'):
            reserve(self.path,'legacy',self.resources,self.rates,self.connection,app_id='app',source_sha256='x')

    def test_oversized_reservation_is_rejected_before_job_exists(self):
        authorize_probe_pool(self.path)
        with self.assertRaisesRegex(ValueError,'budget exhausted'):
            self.reserve(resources={**self.resources,'timeout_s':30000})
        self.assertEqual(json.loads(self.path.read_text())['cloud_probe_pools'][POOL]['jobs'],[])

    def test_no_concurrency_and_failure_does_not_refund_envelope(self):
        authorize_probe_pool(self.path);self.reserve(resources={**self.resources,'timeout_s':19000})
        with self.assertRaisesRegex(ValueError,'One active'):self.reserve('p2')
        record_probe_usage(self.path,'p1',elapsed_s_since_dispatch=10,reported_actual_usd='.01',
                           closed=True,stop_evidence='verified stopped',app_id='app')
        with self.assertRaisesRegex(ValueError,'budget exhausted'):self.reserve('p2')

    def test_runtime_meter_stops_before_reserved_cap_and_counts_only_once(self):
        authorize_probe_pool(self.path);job=self.reserve()
        result=record_probe_usage(self.path,'p1',elapsed_s_since_dispatch=2521,app_id='app')
        self.assertTrue(result['job']['stop_required']);self.assertTrue(result['pool']['stop_required'])
        self.assertEqual(result['pool']['cloud_jobs_dispatched'],1)
        r=json.loads(self.path.read_text());self.assertEqual(r['campaign_committed_upper_usd'],79.)
        self.assertEqual(result['pool']['committed_upper_usd'],job['reserved_upper_usd'])

    def test_late_overrun_is_recorded_and_stops_pool_not_hidden(self):
        authorize_probe_pool(self.path);self.reserve()
        record_probe_usage(self.path,'p1',elapsed_s_since_dispatch=5,closed=True,stop_evidence='stopped',app_id='app')
        result=record_probe_usage(self.path,'p1',elapsed_s_since_dispatch=5,reported_actual_usd='6.01',
                                  billing_evidence_sha256='late-invoice')
        self.assertTrue(result['pool']['cap_breached']);self.assertTrue(result['pool']['stop_required'])
        self.assertEqual(result['job']['status'],'closed')
        self.assertGreater(json.loads(self.path.read_text())['campaign_committed_upper_usd'],79.)


if __name__=='__main__':unittest.main()
