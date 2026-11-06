"""Spawn detached 0.6B training on a deployed app (survives local disconnect)."""
import json
import sys

import modal

app_name = "stage0-cuda-training"
plan = sys.argv[1] if len(sys.argv) > 1 else "/data/600m-native-cf/plan.json"

fn = modal.Function.from_name(app_name, "train_large_plan")
call = fn.spawn(plan)
print(json.dumps({"app": app_name, "call_id": call.object_id, "plan": plan}))
