"""Extract the counterfactual tape tar on the Volume."""
import json

from beatfcn.cloud import train_stage0_app as training

with training.app.run(detach=True):
    print(json.dumps(training.extract_counterfactual_tar.remote(), ensure_ascii=False))
