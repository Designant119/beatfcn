import json
from pathlib import Path
import tempfile
import time
import unittest
from datetime import datetime,timezone

from beatfcn.train.campaign_budget import reserve, update_stage,reconcile_closed_stages


class CampaignBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'ledger.json'
        self.initial = {'paid_stages_dispatched': [], 'planned_stages': {'training': {'reserved_upper_usd': 5.}},
                        'historical_accounted_upper_usd': 4., 'planned_other_reserve_usd': .5,
                        'approved_total_usd': 10.}
        self.path.write_text(json.dumps(self.initial))
        self.resources = {'cpu': 4, 'memory_gib': 16, 'gpu': 'L40S', 'timeout_s': 600}
        self.rates = {'cpu_hour_cost': '.0473', 'mem_gib_hour_cost': '.008', 'gpu_hour_cost_l40s': '1.95'}
        self.connection = {'fresh_check_succeeded': True, 'checked_at_unix': time.time()}

    def call(self, name='diagnostic'):
        return reserve(self.path, name, self.resources, self.rates, self.connection, app_id='app', source_sha256='hash')

    def test_pending_training_counts_and_duplicate_dispatch_rejected(self):
        self.call()
        with self.assertRaisesRegex(ValueError, 'already reserved'):
            self.call()
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            self.call('another')
        self.assertEqual(len(json.loads(self.path.read_text())['paid_stages_dispatched']), 1)

    def test_failed_stage_reservation_is_not_released(self):
        self.call()
        update_stage(self.path, 'diagnostic', status='failed')
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            self.call('retry')

    def test_stale_or_failed_read_does_not_create_reservation(self):
        self.connection['fresh_check_succeeded'] = False
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.call()
        self.assertEqual(json.loads(self.path.read_text()), self.initial)

    def test_planned_stage_replaces_its_pending_envelope(self):
        self.call('training')
        ledger = json.loads(self.path.read_text())
        self.assertLess(ledger['campaign_committed_upper_usd'], 5.)

    def reconcile_fixture(self,state='APP_STATE_STOPPED',recent=False):
        self.call()
        start=(int(time.time()//3600)-(0 if recent else 3))*3600+10
        apps={'checked_at_unix':time.time(),'apps':[dict(app_id='app',ok=True,app_state=state,created_at=start,stopped_at=start+100)]}
        billing=[dict(object_id='app',interval_start=datetime.fromtimestamp(start,timezone.utc).replace(minute=0,second=0).isoformat(),cost='.05')]
        return apps,billing

    def test_reconcile_requires_closed_old_app_billing_and_preserves_reservation(self):
        apps,billing=self.reconcile_fixture()
        before=json.loads(self.path.read_text())['paid_stages_dispatched'][0]['reserved_usd']
        result=reconcile_closed_stages(self.path,apps,billing,self.rates,self.connection,evidence={'lifecycle_sha':'a','billing_sha':'b'})
        stage=json.loads(self.path.read_text())['paid_stages_dispatched'][0]
        self.assertEqual(stage['reserved_usd'],before)
        self.assertGreater(stage['reconciliation']['accounted_upper_usd'],.05)
        self.assertLess(stage['reconciliation']['accounted_upper_usd'],before)
        self.assertEqual(len(result['changes']),1)
        again=reconcile_closed_stages(self.path,apps,billing,self.rates,self.connection,evidence={'lifecycle_sha':'a','billing_sha':'b'})
        self.assertEqual(again['campaign_committed_upper_usd'],result['campaign_committed_upper_usd'])
        with self.assertRaisesRegex(ValueError,'cannot rewrite'):
            update_stage(self.path,'diagnostic',reconciliation={'accounted_upper_usd':0})

    def test_reconcile_retains_unverified_or_recent_reservations(self):
        apps,billing=self.reconcile_fixture(recent=True)
        result=reconcile_closed_stages(self.path,apps,billing,self.rates,self.connection,evidence={'sha':'a'})
        self.assertEqual(result['changes'],[])
        apps['apps'][0]['created_at']-=10800;apps['apps'][0]['stopped_at']-=10800
        apps['apps'][0]['app_state']='APP_STATE_RUNNING'
        self.assertEqual(reconcile_closed_stages(self.path,apps,billing,self.rates,self.connection,evidence={'sha':'a'})['changes'],[])
        apps['apps'][0]['app_state']='APP_STATE_STOPPED'
        self.assertEqual(reconcile_closed_stages(self.path,apps,[],self.rates,self.connection,evidence={'sha':'a'})['changes'],[])

    def test_reconcile_counts_reported_overrun_instead_of_hiding_it(self):
        apps,billing=self.reconcile_fixture();billing[0]['cost']='1.0'
        result=reconcile_closed_stages(self.path,apps,billing,self.rates,self.connection,evidence={'sha':'a'})
        self.assertGreater(result['changes'][0]['after'],1.)


if __name__ == '__main__':
    unittest.main()
