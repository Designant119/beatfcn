"""D42: distinguish member permutation from lost lane/action binding."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import types

import numpy as np
import torch

from beatfcn.model.group_canvas import masked_group_view
from scripts.matched_acoustic_data import make_example
from scripts.r09_local_data import file_sha, verify_files

ROOT = Path(__file__).resolve().parents[1]


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def encoded_content(model, canvas, time_features):
    notes = torch.as_tensor(canvas.notes.reshape(-1, canvas.members, 3))
    lanes = torch.where(notes[..., 2] >= 0, notes[..., 2], 4)
    hold = notes[..., 0] == 1
    tails = (notes[..., 1].float() * 1000 - canvas.layout.start_us) / 1e6
    return (model.kind(notes[..., 0]) + model.lane(lanes) +
            model.tail_time(time_features(tails * hold)) * hold[..., None]).mean(1)


def main():
    start = time.perf_counter()
    plan_path = ROOT / 'bench/rhythm-v2/d42-member-binding-v1.json'
    plan = json.loads(plan_path.read_text())
    verify_files(plan)
    destination = ROOT / plan['result']
    if destination.exists():
        raise FileExistsError('preserve previous audit')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    frozen = types.ModuleType('d42_frozen_p2c')
    frozen.__file__ = str(ROOT / 'scripts/p2c_candidate.py')
    sys.modules[frozen.__name__] = frozen
    exec(compile((ROOT / plan['model_source']).read_bytes(), frozen.__file__, 'exec'), frozen.__dict__)
    torch.manual_seed(plan['seed'])
    contract = json.loads((ROOT / plan['model_contract']).read_text())
    model = frozen.GroupModel(**contract['model']).eval()
    initial = state_sha(model)
    if initial != plan['expected_initial_state_sha256']:
        raise ValueError('R13 initialization not reproduced')
    data_plan = json.loads((ROOT / plan['panel_plan']).read_text())
    manifest = json.loads((ROOT / plan['panel_manifest']).read_text())
    verify_files(manifest)
    records, input_stats = [], {}
    with torch.no_grad():
        for item in manifest['items']:
            if item['row']['mode'] != 'mania_4k':
                continue
            inputs, labels, identity = make_example(item, data_plan)
            eligible = []
            for block, group in zip(*np.nonzero(labels.multiplicity >= 2)):
                n = int(labels.multiplicity[block, group])
                notes = labels.notes[block, group, :n]
                pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                         if not np.array_equal(notes[i, :2], notes[j, :2])]
                if pairs:
                    eligible.append((int(block), int(group), pairs[0]))
            if not eligible:
                continue
            for name in ('local', 'song', 'conditions', 'local_times', 'song_times'):
                value = getattr(inputs, name)
                if not torch.isfinite(value).all():
                    raise ValueError('nonfinite real input')
                input_stats[f"{identity['panel_id']}/{name}"] = dict(shape=list(value.shape),
                    nonfinite=0, minimum=float(value.min()), maximum=float(value.max()),
                    mean=float(value.mean()), std=float(value.std(unbiased=False)))
            audio = model.audio(inputs)
            for block, group, (i, j) in eligible:
                visible = np.zeros_like(labels.multiplicity, dtype=bool)
                visible[block, group] = True
                a = masked_group_view(labels, visible)
                b, control = deepcopy(a), deepcopy(a)
                b.notes[block, group, [i, j], :2] = b.notes[block, group, [j, i], :2]
                held = np.flatnonzero(control.notes[block, group, :, 0] == 1)
                if not len(held):
                    raise ValueError('selected mania actions unexpectedly lack a hold')
                control.notes[block, group, int(held[0]), 1] += 40
                if not np.array_equal(a.notes[..., 2], b.notes[..., 2]) or np.array_equal(a.notes, b.notes):
                    raise ValueError('intervention did not change action binding at fixed lane IDs')
                pooled = [encoded_content(model, view, frozen.time_features) for view in (a, b, control)]
                outputs = [model(inputs, view, audio=audio) for view in (a, b, control)]
                if any(not torch.isfinite(t).all() for output in outputs for t in output.values()):
                    raise ValueError('nonfinite forward')
                delta = {name:float((outputs[0][name] - outputs[1][name]).abs().max()) for name in outputs[0]}
                control_delta = {name:float((outputs[0][name] - outputs[2][name]).abs().max()) for name in outputs[0]}
                records.append(dict(identity=identity, block=block, group=group,
                    changed_member_indices=[i, j], source_group_notes=a.notes[block, group].tolist(),
                    swapped_group_notes=b.notes[block, group].tolist(),
                    pooled_swap_max_abs=float((pooled[0] - pooled[1]).abs().max()),
                    pooled_tail_control_max_abs=float((pooled[0] - pooled[2]).abs().max()),
                    output_swap_max_abs=delta, output_tail_control_max_abs=control_delta))
    if len(records) != plan['expected_pairs'] or state_sha(model) != initial:
        raise ValueError('pair coverage or model immutability failure')
    equivalent = all(max(r['output_swap_max_abs'].values()) <= plan['equivalence_atol'] for r in records)
    nontrivial = all(max(r['output_tail_control_max_abs'].values()) > plan['equivalence_atol'] for r in records)
    result = dict(id='D42', complete=True, algebraic_binding_collision_confirmed=equivalent and nontrivial,
        scope='initialized archived R13, real-chart partial visible drafts; architecture property, no trained-quality claim',
        input_sha256=plan['input_sha256'], plan_sha256=file_sha(plan_path),
        initial_state_sha256=initial, model_unchanged=True,
        all_output_heads_equivalent_for_binding_swaps=equivalent,
        tail_only_negative_control_distinguishable=nontrivial,
        actual_input_stats=input_stats, pairs=records,
        optimizer_updates=0, model_forwards=len(records) * 3,
        model_promotion=False, new_paid_compute_usd=0,
        wall_s=time.perf_counter() - start,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:result[k] for k in ('complete','algebraic_binding_collision_confirmed',
        'model_forwards','optimizer_updates','wall_s','peak_rss_gib')}))


if __name__ == '__main__':
    main()
