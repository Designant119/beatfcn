"""Bounded CPU data/optimizer probe through the actual baseline training driver."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.density_curve import requested_density_curve
from beatfcn.train.local_panel import collate
from beatfcn.train.local_request_panel import event_request
from beatfcn.train.mert_panel import MertRequestPanel
from beatfcn.train.mel_normalization import fit_mel_statistics
from beatfcn.train.training_panel import build_training_panel
from beatfcn.train.control_metrics import density_request_metrics, type_metrics, readout_metrics
from scripts.run_local_song_baseline import run

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/data-contract-v2-final'
CACHE = ROOT/'inputs/data-contract-v2-final-probe'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    if (REPORT/'training.json').exists():
        raise FileExistsError('Probe is single-run; inspect previous evidence before rerunning')
    selection = json.loads((REPORT/'selection.json').read_text())
    tape = json.loads((CACHE/'targets/tape.json').read_text())
    train = dict(tape, charts=[c for c in tape['charts'] if c['split'] == 'train'])
    tape_path = CACHE/'train-tape.json'
    write(tape_path, train)
    for row in selection:
        if row['split'] != 'train':
            continue
        audio = row['audio']
        folder = CACHE/'features'/audio
        folder.mkdir(parents=True, exist_ok=False)
        old = ROOT/'inputs/d96_mert2_features_v1'/audio
        metadata = json.loads((old/'metadata.json').read_text())
        identities = {}
        for name, source in {**{n: old/n for n in ('L12.npy', 'L23.npy', 'times_s.npy')},
                             'mel10_complete.npy': ROOT/row['mel']}.items():
            (folder/name).symlink_to(source)
            identities[name] = dict(sha256=digest(source), bytes=source.stat().st_size)
        write(folder/'receipt.json', dict(complete=True, metadata=metadata, files=identities,
            scope='Local adapter over existing real caches; no encoding or cloud call'))
    features = str(CACHE/'features')
    base = MertRequestPanel(ROOT, str(tape_path), features)
    audios = {r['audio_sha256'] for r in base.records}
    forbidden = {r['audio'] for r in selection if r['split'] != 'train'}
    stats = fit_mel_statistics(base.features, audios, forbidden)
    stats_path = REPORT/'mel-statistics.json'
    write(stats_path, stats)
    indices = []
    for mode in ('phira', 'mania_4k'):
        candidates = [(i, r) for i, r in enumerate(base.records)
            if r['mode'] == mode and r['event_count'] > 0]
        diversity = max(len(r['types']) for _, r in candidates)
        indices.append(min((i for i, r in candidates if len(r['types']) == diversity),
            key=lambda i: base.records[i]['tokens']))
    exposure = dict(pool_identity=hashlib.sha256(json.dumps(base.records,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        steps=[dict(effective_update=i+1, indices=indices*2,
            specified_condition_fields=[[]]*4) for i in range(100)])
    exposure_path = CACHE/'exposure.json'
    write(exposure_path, exposure)
    config = FullEventConfig(d=64, heads=4, kv_heads=2, ffn=128, encoder_layers=1,
        decoder_layers=2, taps=2, bands=1, band_slots=1, mel_bins=128, intensity_dim=3,
        dropout=0., qk_norm=True, learnable_audio_scale=True)
    plan = dict(id='data-contract-v2-cpu-probe', data_contract_version=2,
        tape=str(tape_path.relative_to(ROOT)), mert_features=features,
        features='reports/d94_merged_feature_admission_v1.json',
        exposure=str(exposure_path.relative_to(ROOT)), steps=100, device='cpu', cpu_threads=2,
        seed=20261101, config=asdict(config), physical_microbatch_requests=4,
        learning_rate=.005, optimizer=dict(weight_decay=0., betas=[.9, .95]),
        learning_rate_schedule=dict(warmup_updates=2, minimum_lr=.001),
        loss=dict(type_weight=.1, z_weight=1e-4), gradient_diagnostics=True,
        intensity=dict(cell_us=100_000, scale=16., hide_probability=0.),
        mel_normalization=dict(path=str(stats_path.relative_to(ROOT)), sha256=digest(stats_path)),
        health='reports/data-contract-v2-final/preflight.json',
        output='reports/data-contract-v2-final/training.json', checkpoint_dir='inputs/data-contract-v2-final-probe/checkpoints',
        tensor_probe_indices=indices, validation_indices=[], checkpoints=[], checkpoint_every=100,
        skip_initial_eval=True, skip_generation=True, max_train_wall_s=300,
        prefetch_workers=1, prefetch_lookahead=2)
    panel = build_training_panel(ROOT, plan)
    x, y = collate([panel.row(i) for i in indices])
    preflight = dict(passed=all(bool(torch.isfinite(v).all()) for section in (x, y) for v in section.values()),
        tensors={name: dict(shape=list(v.shape), finite=bool(torch.isfinite(v).all())) for name,v in x.items()},
        selected_indices=indices, events=[panel.records[i]['event_count'] for i in indices],
        selected_types=[panel.records[i]['types'] for i in indices], valid_tokens=int(y['valid'].sum()),
        heldout_audio=list(forbidden), independent_mert_validation_ready=False,
        limitation='Both frozen validation songs lack local MERT caches; not relabeled as train')
    write(REPORT/'preflight.json', preflight)
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed'])
    model = FullEventModel(config).eval()
    with torch.no_grad():
        initial_ce = float(event_loss(model(x, y), y))
    del model
    identities = [tape_path, exposure_path, stats_path, REPORT/'preflight.json']
    identities += [CACHE/'features'/a/'receipt.json' for a in audios]
    plan['inputs_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in identities}
    source_paths = [Path(__file__), ROOT/'scripts/run_local_song_baseline.py']
    source_paths += list((ROOT/'beatfcn/train').glob('*.py'))
    source_paths += list((ROOT/'beatfcn/model').glob('*.py'))
    plan['code_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    write(REPORT/'probe-plan.json', plan)
    tick = time.monotonic()
    run(REPORT/'probe-plan.json')
    report = json.loads((REPORT/'training.json').read_text())
    checkpoint = ROOT/report['checkpoint']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = FullEventModel(config).eval()
    model.load_state_dict(saved['model'])
    with torch.no_grad():
        final_ce = float(event_loss(model(x, y), y))
    outputs = []
    index = indices[0]
    request = event_request(panel.records[index]['request'])
    for density in (0., 32.):
        segments = [dict(start_us=request.start_us, end_us=request.end_us, density=density)]
        row = panel.row(index)
        row['inputs'].update(requested_density_curve(segments, request.start_us, request.end_us))
        inp, _ = collate([row])
        generated = generate_batch(model, inp, [request], max_tokens=256,
            deadline=time.monotonic()+30)[0]
        outputs.append(dict(requested_density=density, prediction=generated,
            curve=density_request_metrics(generated, segments),
            types=type_metrics(generated, panel.targets(index)), readout=readout_metrics([generated]),
            scope='Training-song counterfactual diagnostic; type counts are not matched recall'))
    result = dict(updates=report['optimizer_updates'], initial_joint_ce=initial_ce,
        final_joint_ce=final_ce, implementation_health_passed=final_ce <= .05,
        preclip_gradient_norm=report['gradient_norm'],
        per_layer_preclip_norm=report['per_layer_preclip_norm'],
        wall_s=time.monotonic()-tick, weights_saved=False, cloud_spend_usd=0,
        free_control_diagnostics=outputs, independent_song_quality_tested=False,
        training_parameters=sum(p.numel() for p in model.parameters()))
    write(REPORT/'probe-result.json', result)
    if digest(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Owned health checkpoint changed before disposal')
    checkpoint.unlink()
    print(json.dumps({k:v for k,v in result.items() if k not in ('free_control_diagnostics','per_layer_preclip_norm')}))


if __name__ == '__main__':
    main()
