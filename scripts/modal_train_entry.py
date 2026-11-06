"""Modal CLI entrypoint for detached 0.6B training."""
import json

import modal

from beatfcn.cloud import train_stage0_app as training

app = training.app


@app.local_entrypoint()
def main(plan: str):
    result = training.train_large_plan.remote(plan)
    print(json.dumps(result, ensure_ascii=False, indent=1))
