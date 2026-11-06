"""Rebuild the 580 native tape using MERT receipt durations."""
import json

from beatfcn.cloud import train_stage0_app as training

with training.app.run(detach=True):
    print(json.dumps(training.rebuild_native_mert_tape.remote(), ensure_ascii=False))
