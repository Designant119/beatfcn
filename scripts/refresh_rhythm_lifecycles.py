"""Read exact paid-app lifetimes for conservative campaign reconciliation."""
import asyncio
import json
from pathlib import Path
import time

from modal.client import _Client
from modal_proto import api_pb2

ROOT=Path(__file__).resolve().parents[1]


async def refresh():
    ledger=json.loads((ROOT/'reports/rhythm_v2_budget.json').read_text())
    apps=sorted({s['app_id'] for s in ledger['paid_stages_dispatched']})
    client=await _Client.from_env();semaphore=asyncio.Semaphore(4);results={}
    async def read(app_id):
        try:
            async with semaphore:
                response=await asyncio.wait_for(client._stub.AppGetLifecycle(
                    api_pb2.AppGetLifecycleRequest(app_id=app_id)),20)
            life=response.lifecycle
            results[app_id]=dict(app_id=app_id,ok=True,app_state=api_pb2.AppState.Name(life.app_state),
                created_at=life.created_at,stopped_at=life.stopped_at)
        except Exception as exc:
            results[app_id]=dict(app_id=app_id,ok=False,error=f'{type(exc).__name__}: {exc}')
    try:
        await asyncio.wait_for(asyncio.gather(*(read(app_id) for app_id in apps)),50)
    except TimeoutError:
        for app_id in apps:
            results.setdefault(app_id,dict(app_id=app_id,ok=False,error='overall read deadline'))
    report=dict(checked_at_unix=time.time(),read_only=True,apps=[results[x] for x in apps])
    path=ROOT/'reports/rhythm_v2_stage_lifecycles.json';temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
    print(json.dumps(dict(apps=len(apps),read_successes=sum(x['ok'] for x in results.values()),
        stopped=sum(x.get('app_state')=='APP_STATE_STOPPED' for x in results.values()))))


if __name__=='__main__':asyncio.run(refresh())
