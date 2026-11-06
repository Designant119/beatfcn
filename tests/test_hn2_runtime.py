"""CPU gate/schedule tests. No model construction, CUDA or Modal operations."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from beatfcn.train.hn2_runtime import (
    GRANT,REQUIRED_SOURCES,ConstructionWatchdog,ResourceGuard,RuntimeConfig,
    assert_public_inputs,check_fresh_connection,evaluation_due,local_launch_gate,
    run_evaluation_schedule,sha,validate_lifecycle_costs,zero_audio_inputs,diagnostic_context_states,
)


class ResourceTests(unittest.TestCase):
    def test_original_thresholds_and_exact_wait_boundary(self):
        guard=ResourceGuard()
        guard.observe(70,elapsed_s=480,data_wait_s=384)
        with self.assertRaisesRegex(RuntimeError,'80%'):
            guard.observe(90,elapsed_s=480,data_wait_s=384.001)
        with self.assertRaisesRegex(RuntimeError,'70%'):
            ResourceGuard().observe(69.99,elapsed_s=480,data_wait_s=0)
        ResourceGuard().observe(0,elapsed_s=479.99,data_wait_s=0)
        with self.assertRaises(ValueError):RuntimeConfig(min_util=69)

    def test_latest_eight_unweighted_and_missing_observation_stops(self):
        guard=ResourceGuard()
        for i in range(8):guard.observe(0 if i==0 else 70,elapsed_s=i,data_wait_s=0)
        self.assertEqual(guard.observe(70,elapsed_s=480,data_wait_s=0)['util_rolling_mean'],70)
        for value in (None,float('nan'),float('inf')):
            with self.assertRaisesRegex(RuntimeError,'observation'):
                ResourceGuard().observe(value,elapsed_s=1,data_wait_s=0)

    def test_real250_schedule_uses_applied_updates_and_endpoint(self):
        self.assertEqual([i for i in range(1,1001) if evaluation_due(i)],[250,500,750,1000])
        self.assertFalse(evaluation_due(0))
        self.assertTrue(evaluation_due(260,RuntimeConfig(applied_updates=260)))

    def test_construction_watchdog_detects_app_before_context_yield_and_keeps_waiting(self):
        watchdog=ConstructionWatchdog(100,20)
        self.assertFalse(watchdog.tick(None,79)['stop_required'])
        self.assertTrue(watchdog.tick(None,80)['stop_required'])
        late=watchdog.tick('ap-owned',81)
        self.assertTrue(late['stop_required']);self.assertTrue(late['meter_due'])
        self.assertEqual(late['app_id'],'ap-owned')
        self.assertTrue(watchdog.tick(None,100)['hard_stop_required'])
        with self.assertRaises(RuntimeError):watchdog.tick('ap-other',101)

    def test_free_phase_persists_before_any_proper_and_failed_denominator_remains(self):
        calls=[];saved=[];specs=[{'id':i} for i in range(3)]
        def free(spec):calls.append(('free',spec['id']));return {'machine_z':spec['id']}
        def proper(spec,result):calls.append(('proper',spec['id']));return result
        outcomes=run_evaluation_schedule(specs,free,proper,lambda phase,rows:saved.append((phase,len(rows))))
        self.assertEqual(calls,[('free',0),('free',1),('free',2),('proper',0),('proper',1),('proper',2)])
        self.assertTrue(all(n==3 for _,n in saved));self.assertEqual(len(outcomes),3)
        def fail(spec):raise RuntimeError('decode')
        saved=[]
        with self.assertRaises(RuntimeError):run_evaluation_schedule(specs,fail,proper,lambda p,r:saved.append((p,len(r))))
        self.assertEqual(saved,[('expected',3)])

    def test_zero_control_all_audio_streams_and_label_isolation(self):
        x={k:torch.ones(2,3) for k in ('local','mel','song','whole_mel','whole_mel_times','head_us')}
        zero=zero_audio_inputs(x)
        for key in ('local','mel','song','whole_mel'):self.assertEqual(float(zero[key].sum()),0)
        self.assertTrue(torch.equal(zero['whole_mel_times'],x['whole_mel_times']))
        self.assertTrue(torch.equal(zero['head_us'],x['head_us']))
        arrays={k:np.ones((2,3),np.float32) for k in ('local','mel','song','whole_mel','whole_mel_times')}
        numpy_zero=zero_audio_inputs(arrays)
        self.assertEqual(float(numpy_zero['whole_mel'].sum()),0)
        np.testing.assert_array_equal(numpy_zero['whole_mel_times'],arrays['whole_mel_times'])
        with self.assertRaises(ValueError):assert_public_inputs({**x,'targets':torch.ones(1)})

    def test_context_controls_keep_fixed_machine_and_mark_true_as_oracle(self):
        events=torch.zeros(1,512,4,dtype=torch.int64);valid=torch.zeros(1,512,dtype=torch.bool)
        events[0,0]=torch.tensor([1000,3000,1,-1]);valid[0,0]=True
        machine=dict(events=events.clone()+7,active=torch.ones_like(valid))
        controls=diagnostic_context_states(machine,events,valid)
        self.assertIs(controls['own_z'],machine)
        self.assertFalse(controls['null']['active'].any())
        self.assertEqual(int(controls['true_oracle']['active'].sum()),1)
        self.assertTrue(torch.equal(controls['true_oracle']['events'][0,0],events[0,0]))
        self.assertTrue(torch.equal(machine['events'],events+7))


class LaunchGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        def put(name,value):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value) if not isinstance(value,str) else value)
        self.put=put
        for name in REQUIRED_SOURCES:put(name,'# frozen\n')
        sdk_names=['.venv/lib/python3.12/site-packages/modal/'+name for name in ('app.py','runner.py')]
        for name in sdk_names:put(name,'# pre-build app id evidence\n')
        sources={name:sha(self.root/name) for name in REQUIRED_SOURCES}
        put('data.json',dict(complete=True,target_slots=512,records=[dict(array_index=1,split='dev')]))
        put('upload.json',dict(files={}))
        put('prior.json',dict(duration={}))
        data_sha={name:sha(self.root/name) for name in ('data.json','upload.json','prior.json')}
        common=dict(passed=True,candidate_id='candidate',source_sha256=sources,data_manifest_sha256=data_sha)
        put('independent.md','Independent original')
        put('review.json',dict(common,independent_original_reports=[dict(path='independent.md',sha256=sha(self.root/'independent.md'))]))
        put('local.json',common)
        put('ledger.json',dict(conditional_candidate_authorizations={GRANT:dict(owner_authorized=True,new_usd_cap='8')}))
        put('connection.json',dict(fresh_check_succeeded=True,checked_at_unix=1000,
            evidence_sha256={'apps':'a','rates':'b','billing':'c'},full_lifecycle_bound_verified=True))
        self.rates=dict(gpu_usd_s='.001',cpu_usd_s='.00001',memory_gib_usd_s='.000002')
        put('rates.json',self.rates)
        self.contract=dict(candidate_id='candidate',grant_id=GRANT,actual_trainable_parameters=138621056,
            local_modal_sdk_sha256={name:sha(self.root/name) for name in sdk_names},
            source_sha256=sources,data_manifest_sha256=data_sha,data_manifest='data.json',upload_manifest='upload.json',
            duration_prior_path='prior.json',independent_review_receipt='review.json',local_gate_receipt='local.json',
            ledger_path='ledger.json',connection_path='connection.json',rates_path='rates.json',
            density_rule='all-events-1s-gt12-drag-only-v2',evaluate_every=250,
            evaluation_panel=[dict(array_index=1,audio='real')],
            cloud_health=dict(updates=100,effective_batch=5,fit_draft_indices=[1,2,3,4,5],
                acceptance=dict(maximum_total_loss=1,maximum_loss_ratio=.5,maximum_field_ratio=1.05)),
            resources=dict(gpu='H100',max_containers=1,retries=0,cpu=12,memory_gib=64,timeout_s=3600,
                startup_allowance_s=120,stop_allowance_s=60,other_upper_usd=2),
            lifecycle=dict(cpu_asset_verification=dict(containers=1,cpu=4,memory_gib=4,timeout_s=1200),
                image_build_upper_usd=.1,upload_upper_usd=.1,failure_cleanup_upper_usd=.1,
                volume_storage=dict(gib=20,retention_days=1,upper_usd=.1,cleanup_plan='Owned-prefix export and verified cleanup'),
                egress=dict(gib=20,upper_usd=.1)))
        put('contract.json',self.contract)

    def tearDown(self):self.tmp.cleanup()

    def test_valid_frozen_local_checks_have_no_modal_import(self):
        with patch.dict('sys.modules',{'modal':None}):
            result=local_launch_gate(self.root,'contract.json',now=1000)
        self.assertEqual(result['authorization']['cap_usd'],'8')

    def test_pending_cap_and_source_review_drift_fail_before_any_cloud(self):
        self.put('ledger.json',dict(conditional_candidate_authorizations={GRANT:dict(owner_authorized=False,new_usd_cap=None)}))
        with patch.dict('sys.modules',{'modal':None}):
            with self.assertRaisesRegex(ValueError,'cap'):local_launch_gate(self.root,'contract.json',now=1000)
        self.put('ledger.json',dict(conditional_candidate_authorizations={GRANT:dict(owner_authorized=True,new_usd_cap='8')}))
        self.put(REQUIRED_SOURCES[0],'# drift\n')
        with self.assertRaisesRegex(ValueError,'source drift'):local_launch_gate(self.root,'contract.json',now=1000)

    def test_stale_billing_and_missing_extra_lifecycle_cost_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'600s'):local_launch_gate(self.root,'contract.json',now=1600)
        contract=deepcopy(self.contract);contract['resources']['other_upper_usd']=0
        with self.assertRaisesRegex(ValueError,'other_upper_usd'):validate_lifecycle_costs(contract,self.rates)
        connection=dict(fresh_check_succeeded=True,checked_at_unix=1000,evidence_sha256={'a':'b'})
        with self.assertRaises(ValueError):check_fresh_connection(connection,now=1001)


if __name__=='__main__':unittest.main()
