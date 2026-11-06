"""C20-S failure injections: deadlines must survive blocked accounting/logs."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import fcntl
import asyncio
from types import SimpleNamespace

from beatfcn.train.hn2_stop_supervisor import OwnedStopSupervisor
from beatfcn.train.bounded_ledger import locked_ledger
from beatfcn.train.hn2_contract import contract_digest
from beatfcn.train.hn2_safe_runtime import DeadlineWatchdog


class SupervisorTests(unittest.TestCase):
    def run_blocked(self,late=False):
        ledger_release=threading.Event();log_release=threading.Event();stopped=threading.Event();entered=threading.Event()
        app={'id':None if late else 'ap-owned'}
        def record(app_id,proof):
            entered.set();ledger_release.wait();return dict(job=dict(stop_required=False))
        def publish(state):log_release.wait()
        ctl=OwnedStopSupervisor(time.time()+.10,.05,app_id=lambda:app['id'],cancel=lambda:None,
            stop_app=lambda ident:stopped.set(),verify=lambda ident:{'state':'stopped','tasks':0},
            record=record,cleanup=lambda:{'passed':True},publish=publish,
            interval=.002,record_interval=.005,stop_interval=.005).start()
        try:
            if late:
                time.sleep(.06);app['id']='ap-owned'
            self.assertTrue(entered.wait(.3))
            self.assertTrue(stopped.wait(.3),'Stop may not wait on ledger or log')
            self.assertFalse(ledger_release.is_set());self.assertFalse(log_release.is_set())
        finally:
            ledger_release.set();log_release.set();ctl.request_stop('test cleanup')
            self.assertTrue(ctl.finished.wait(1));ctl.clock.join(1)

    def test_blocked_accounting_and_logging_do_not_freeze_stop(self):self.run_blocked()
    def test_build_period_late_app_id_still_stops(self):self.run_blocked(late=True)

    def test_internal_hard_timer_does_not_wait_on_soft_log(self):
        for raises in (False,True):
            release=threading.Event();hard=threading.Event()
            def soft():
                if raises:raise OSError('log')
                release.wait()
            watch=DeadlineWatchdog(time.time()+.05,.03,on_soft_stop=soft,hard_stop=hard.set).start()
            try:self.assertTrue(hard.wait(.3))
            finally:release.set();watch.close();watch.thread.join(.3)

    def test_actual_flock_acquisition_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger.json';path.write_text('{}')
            with path.with_suffix('.lock').open('a') as held:
                fcntl.flock(held,fcntl.LOCK_EX)
                start=time.monotonic()
                with self.assertRaises(TimeoutError):
                    with locked_ledger(path,timeout_s=.04):pass
                self.assertLess(time.monotonic()-start,.3)

    def test_complete_contract_hash_covers_every_execution_dimension(self):
        base=dict(model={'local':4},resources={'timeout':1800},health={'threshold':.5},
            evaluation=[1,2],prior='a',source={'a':'b'},seed=7)
        original=contract_digest(base)
        for key in base:
            changed={**base,key:None};self.assertNotEqual(contract_digest(changed),original,key)
        self.assertEqual(contract_digest({**base,'contract_sha256':original}),original)
        with self.assertRaises(ValueError):contract_digest({'acceptance':float('inf')})

    def test_stopped_worker_output_recovered_before_any_deletion(self):
        from beatfcn.train.hn2_output_recovery import recover_output
        class Missing(Exception):pass
        async def listing(path,recursive):
            return [SimpleNamespace(path='/runs/owned/phase.json',type=SimpleNamespace(name='FILE'))]
        async def read(path):yield b'{"quality_updates":0}'
        volume=SimpleNamespace(listdir=SimpleNamespace(aio=listing),read_file=SimpleNamespace(aio=read))
        result=asyncio.run(recover_output(volume,'owned',Missing))
        self.assertEqual(result['phase.json']['text'],'{"quality_updates":0}')


if __name__=='__main__':unittest.main()
