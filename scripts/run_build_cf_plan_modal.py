"""Build the 0.6B counterfactual plan on the Modal CPU container."""
import json

from beatfcn.cloud import train_stage0_app as training

with training.app.run():
    result = training.build_counterfactual_plan.remote(
        '/data/counterfactual-d94-580-v1/tape-modal.json',
        '/data/counterfactual-d94-580-v1/curves.json',
        '/data/data-contract-v2-mel.json',
        '600m-cf-580', 20000)
print(json.dumps(result, ensure_ascii=False, indent=1))
