"""Build the 0.6B native-tape counterfactual plan on Modal CPU."""
import json

from beatfcn.cloud import train_stage0_app as training

with training.app.run(detach=True):
    result = training.build_native_counterfactual_plan.remote(
        '/data/native-v2-d94-580-mert/tape.json',
        '/data/data-contract-v2-mel.json',
        '600m-native-cf', 2000, 16)  # optimizer updates, accum steps
print(json.dumps(result, ensure_ascii=False, indent=1))
