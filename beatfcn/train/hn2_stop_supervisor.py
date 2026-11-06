"""Deadline and actual owned stop are independent of accounting/logging IO."""
import threading
import time


class OwnedStopSupervisor:
    def __init__(self,deadline,allowance,*,app_id,cancel,stop_app,verify,record,cleanup,publish,
                 interval=1.,record_interval=10.,stop_interval=5.):
        self.deadline=deadline;self.soft=deadline-allowance;self.app_id=app_id
        self.cancel=cancel;self.stop_app=stop_app;self.verify=verify;self.record=record
        self.cleanup=cleanup;self.publish=publish;self.interval=interval
        self.record_interval=record_interval;self.stop_interval=stop_interval
        self.stop=threading.Event();self.finished=threading.Event()
        self.state=dict(app_id=None,stop_verified=False,accounting_closed=False,cleanup_passed=False,reason=None)
        self.clock=threading.Thread(target=self._clock,daemon=False)

    def start(self):
        threading.Thread(target=self._account,daemon=True).start()
        threading.Thread(target=self._publish,daemon=True).start()
        self.clock.start();return self

    def request_stop(self,reason):
        self.state['reason']=self.state['reason'] or reason;self.stop.set()

    def check(self):
        if self.stop.is_set() or time.time()>=self.soft:raise TimeoutError(self.state['reason'] or 'Lifecycle stop allowance')

    @staticmethod
    def _background(callback):
        def invoke():
            try:callback()
            except BaseException:pass
        threading.Thread(target=invoke,daemon=True).start()

    def _clock(self):
        last=-float('inf');cancelled=False;cleanup_started=False
        while not self.finished.is_set():
            # No ledger, log, user callback, lock or RPC runs on this thread.
            try:app_id=self.app_id()
            except BaseException:
                self.request_stop('App identity getter failed');app_id=self.state['app_id']
            if app_id:
                if self.state['app_id'] not in (None,app_id):self.request_stop('Owned app identity changed')
                else:self.state['app_id']=app_id
            if time.time()>=self.soft:self.request_stop('Independent deadline')
            if self.stop.is_set() and self.state['app_id']:
                if not cancelled:self._background(self.cancel);cancelled=True
                if not self.state['stop_verified'] and time.time()-last>=self.stop_interval:
                    last=time.time();self._background(self._stop_once)
                if self.state['stop_verified'] and not cleanup_started:
                    cleanup_started=True;self._background(self._cleanup)
            if self.state['stop_verified'] and self.state['accounting_closed'] and self.state['cleanup_passed']:
                self.finished.set();return
            self.finished.wait(self.interval)

    def _stop_once(self):
        app_id=self.state['app_id']
        try:
            self.stop_app(app_id)  # caller supplies a bounded subprocess RPC
            proof=self.verify(app_id)
            if proof is not None:self.state.update(stop_verified=True,stop_evidence=proof)
        except BaseException as exc:self.state['stop_error']=type(exc).__name__

    def _account(self):
        while not self.finished.is_set():
            if self.state['app_id'] and not self.state['accounting_closed']:
                try:
                    proof=self.state.get('stop_evidence');value=self.record(self.state['app_id'],proof)
                    self.state['usage']=value
                    if proof is not None:self.state['accounting_closed']=True
                    if value['job']['stop_required']:self.request_stop('Budget cutoff')
                except BaseException as exc:
                    self.state['accounting_error']=type(exc).__name__;self.request_stop('Accounting failure')
            self.finished.wait(self.record_interval)

    def _cleanup(self):
        while not self.finished.is_set() and not self.state['cleanup_passed']:
            try:
                result=self.cleanup();self.state['cleanup']=result
                self.state['cleanup_passed']=result.get('passed') is True
            except BaseException as exc:self.state['cleanup_error']=type(exc).__name__
            if not self.state['cleanup_passed']:self.finished.wait(self.stop_interval)

    def _publish(self):
        while not self.finished.is_set():
            try:self.publish(dict(self.state))
            except BaseException:self.request_stop('Status write failed')
            self.finished.wait(self.interval)
