"""Control-flow and stratified gates for discarded CUDA health;no NN/cloud."""
import ast
import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from beatfcn.train.hn2_admission_runtime import (
    assess_stratified_health,settle_owned_stop,validate_envelope,
    cleanup_owned_files,collect_health_evidence,cleanup_remote_prefixes,
    execution_scope,local_launch_gate,REQUIRED_SOURCES,sha,
)


class StopTests(unittest.TestCase):
    def test_actual_stop_and_ledger_precede_safe_return(self):
        events=[]
        result=settle_owned_stop('ap-owned',cancel=lambda:events.append('cancel'),
            stop=lambda app:events.append(('stop',app)),
            verify=lambda app:(events.append(('verify',app)) or {'state':'stopped','tasks':0}),
            record=lambda app,evidence:(events.append(('record_closed',bool(evidence))) or {'closed':True}))
        self.assertEqual(events,['cancel',('stop','ap-owned'),('verify','ap-owned'),('record_closed',True)])
        self.assertTrue(result['safe_to_exit'])

    def test_failed_cancellation_does_not_skip_stop_or_reconcile(self):
        events=[]
        def cancel():events.append('cancel');raise OSError('transport')
        result=settle_owned_stop('ap-owned',cancel=cancel,stop=lambda app:events.append('stop'),
            verify=lambda app:None,record=lambda app,evidence:(events.append(('record_open',evidence)) or {}))
        self.assertEqual(events,['cancel','stop',('record_open',None)])
        self.assertFalse(result['safe_to_exit']);self.assertFalse(result['stop_verified'])
        self.assertEqual(result['errors'],['cancel:OSError'])

    def test_no_app_id_cannot_claim_closed(self):
        def forbidden(*args):raise AssertionError('No owned app')
        self.assertFalse(settle_owned_stop(None,cancel=forbidden,stop=forbidden,
            verify=forbidden,record=forbidden)['safe_to_exit'])


class StratifiedTests(unittest.TestCase):
    def setUp(self):
        def row(index,after=False):
            fields=dict(occupancy=.3 if after else 1.,head=4. if after else 10.,kind=.4 if after else 1.,tail=4. if after else 10.)
            types={k:dict(n=n,mean_nll=.4 if after else 1.) for k,n in [('tap',10),('hold',8),('flick',6),('drag',1)]}
            return dict(draft_index=index,members=0 if index==4 else 25,holds=0 if index==4 else 8,
                supported=True,p0=dict(fields=fields,types=types,expected=.5),
                p1=dict(fields=deepcopy(fields),types=deepcopy(types),expected=.5))
        self.before=dict(loss=26.,rows=[row(i) for i in range(5)])
        self.after=dict(loss=6.,rows=[row(i,True) for i in range(5)])
        self.free=[dict(draft_index=i,p0=dict(predicted=0 if i==4 else 25),p1=dict(predicted=0 if i==4 else 25)) for i in range(5)]
        self.acceptance=dict(maximum_total_loss=6.,maximum_loss_ratio=.30,maximum_field_ratio=.5)

    def verdict(self):
        return assess_stratified_health(self.before,self.after,self.free,[100]*196,self.acceptance)

    def test_valid_research_fit_is_not_a_product_f1_gate(self):
        self.assertTrue(self.verdict()['passed'])

    def test_single_rare_drag_regression_cannot_hide_in_pooled_loss(self):
        self.after['rows'][0]['p0']['types']['drag']['mean_nll']=1.001
        self.assertFalse(self.verdict()['passed'])

    def test_per_window_head_tail_and_positive_type_gates(self):
        for field in ('head','tail'):
            original=self.after['rows'][1]['p0']['fields'][field]
            self.after['rows'][1]['p0']['fields'][field]=5.001
            self.assertFalse(self.verdict()['passed'],field)
            self.after['rows'][1]['p0']['fields'][field]=original
        self.after['rows'][1]['p0']['types']['tap']['mean_nll']=.501
        self.assertFalse(self.verdict()['passed'])

    def test_empty_must_be_silent_in_both_passes_and_expected_at_most_one(self):
        for p in ('p0','p1'):
            self.free[4][p]['predicted']=1;self.assertFalse(self.verdict()['passed'])
            self.free[4][p]['predicted']=0
        self.after['rows'][4]['p0']['expected']=1.001
        self.assertFalse(self.verdict()['passed'])

    def test_cannot_relax_threshold_or_change_denominator(self):
        self.acceptance['maximum_total_loss']=6.001
        with self.assertRaises(ValueError):self.verdict()
        self.acceptance['maximum_total_loss']=6.
        self.after['rows'][0]['p0']['types']['drag']['n']=0
        with self.assertRaises(ValueError):self.verdict()


class ScopeTests(unittest.TestCase):
    def test_quality_contract_is_rejected_before_any_other_gate(self):
        with self.assertRaisesRegex(ValueError,'health only'):
            validate_envelope({'execution_stage':'quality'}, {}, {}, {})

    def test_review_cannot_be_reused_for_changed_scientific_scope(self):
        with self.assertRaisesRegex(ValueError,'exact scientific'):
            validate_envelope(dict(execution_stage='cuda_health_only',quality_updates_allowed=0),{},dict(execution_scope={}),{})

    def test_health_entry_has_no_quality_optimizer_or_hard_exit(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'beatfcn/train/hn2_admission_runtime.py').read_text()
        tree=ast.parse(source);entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_cuda')
        calls=[n for n in ast.walk(entry) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in calls))
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='HN2Native' for n in calls),1)
        self.assertTrue(any(isinstance(n,ast.Return) for n in ast.walk(entry)))
        launcher=(root/'scripts/run_hn2_admission.py').read_text()
        for text in [source,launcher]:
            t=ast.parse(text)
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_exit' for n in ast.walk(t)))
        self.assertIn('threading.Thread(target=meter,daemon=False)',launcher)

    def test_real_local_gate_needs_no_modal_and_locks_health_scope(self):
        from tests.test_hn2_runtime import LaunchGateTests
        import json
        fixture=LaunchGateTests();fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup)
        c=fixture.contract
        for name in REQUIRED_SOURCES:fixture.put(name,'# frozen\n')
        c.update(execution_stage='cuda_health_only',quality_updates_allowed=0,run_id='c21-health-1',
            source_sha256={name:sha(fixture.root/name) for name in REQUIRED_SOURCES})
        c['cloud_health']['fit_draft_indices']=[1736,455,3242,1770,1859]
        c['cloud_health']['acceptance']=dict(maximum_total_loss=6,maximum_loss_ratio=.3,maximum_field_ratio=.5)
        for name in ('review.json','local.json'):
            r=json.loads((fixture.root/name).read_text());r['source_sha256']=c['source_sha256']
            r['execution_scope']=execution_scope(c);fixture.put(name,r)
        fixture.put('contract.json',c)
        with patch.dict('sys.modules',{'modal':None}):
            value=local_launch_gate(fixture.root,'contract.json',now=1000)
        self.assertEqual(value['contract']['execution_stage'],'cuda_health_only')
        c['quality_updates_allowed']=1000;fixture.put('contract.json',c)
        with self.assertRaises(ValueError):local_launch_gate(fixture.root,'contract.json',now=1000)


class CleanupTests(unittest.TestCase):
    def test_only_exact_owned_prefixes_removed_and_evidence_is_packed_first(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('hn2-assets/owned/data.npy','runs/owned/health.json','runs/other/keep.json'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            packed=collect_health_evidence(root/'runs/owned')
            self.assertIn('health.json',packed)
            self.assertTrue(cleanup_owned_files(root,'owned')['passed'])
            self.assertTrue((root/'runs/other/keep.json').is_file())
            with self.assertRaises(ValueError):cleanup_owned_files(root,'../other')

    def test_binary_health_weights_are_never_silently_discarded(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'model.pt').write_bytes(b'not permitted')
            with self.assertRaisesRegex(ValueError,'tensor/binary'):collect_health_evidence(temp)

    def test_remote_fallback_uses_only_two_owned_prefixes(self):
        calls=[]
        class Missing(Exception):pass
        async def remove(path,recursive):calls.append(('rm',path,recursive))
        async def listing(path):calls.append(('ls',path));raise Missing()
        volume=SimpleNamespace(remove_file=SimpleNamespace(aio=remove),listdir=SimpleNamespace(aio=listing))
        result=asyncio.run(cleanup_remote_prefixes(volume,'owned',Missing))
        self.assertTrue(result['passed'])
        self.assertEqual(calls,[('rm','/hn2-assets/owned',True),('ls','/hn2-assets/owned'),
            ('rm','/runs/owned',True),('ls','/runs/owned')])


if __name__=='__main__':unittest.main()
