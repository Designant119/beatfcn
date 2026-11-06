"""Freeze a positive-pair cohort and prospective validation selection, before inference."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from beatfcn.train.data import selected_for_training

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = ROOT/'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl'
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    by_id = {r['chart_id']:r for r in rows}
    prior_paths = ['bench/capability-v1/manifest.json','reports/r07b_evaluate.json']
    excluded_audio, excluded_ids = set(), set()
    def collect(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ('audio_sha256','audio_id') and isinstance(v,str): excluded_audio.add(v)
                if k=='chart_id' and isinstance(v,str): excluded_ids.add(v)
                collect(v)
        elif isinstance(x,list):
            for v in x: collect(v)
    for p in prior_paths: collect(json.loads((ROOT/p).read_text()))
    excluded_audio.update(by_id[c]['audio_sha256'] for c in excluded_ids if c in by_id)
    support_path = ROOT/'reports/d09_real_pair_support.json'
    support = json.loads(support_path.read_text())
    train = [r for r in support['records'] if r['changed_fields'] and not r['rhythm_targets_equal']]
    train_audio = {r['audio_sha256'] for r in train}
    train_ids = {c for r in train for c in r['chart_ids']}
    if not all(by_id[c]['split']=='train' for c in train_ids): raise ValueError('training split differs')
    edges, buckets = [0,2,4,6,8,12,float('inf')], ['0–2','2–4','4–6','6–8','8–12','12+']
    candidates = defaultdict(list)
    for r in rows:
        if r['split']!='val' or not r.get('kept') or not selected_for_training(r): continue
        if r['audio_sha256'] in excluded_audio|train_audio: continue
        density = r['n_events']/r['audio_sec']
        i = next(i for i in range(6) if edges[i]<=density<edges[i+1])
        candidates[(r['mode'],buckets[i])].append(r)
    validation, missing, used = [], [], set()
    for mode in ['mania_4k','phira']:
        for bucket in buckets:
            choices = sorted(candidates[(mode,bucket)],
                key=lambda r:hashlib.sha256(('r09-independent-v1:'+r['chart_id']).encode()).hexdigest())
            row = next((r for r in choices if r['audio_sha256'] not in used),None)
            if row is None:
                missing.append(dict(mode=mode,bucket=bucket,reason='No eligible unused validation audio'))
                continue
            used.add(row['audio_sha256'])
            validation.append(dict(row=row,selection_bucket=bucket,
                selection_density=row['n_events']/row['audio_sec']))
    sources = [str(manifest.relative_to(ROOT)),str(support_path.relative_to(ROOT)),*prior_paths,
        'beatfcn/train/condition_pairs.py',str(Path(__file__).relative_to(ROOT))]
    plan = dict(schema='beatfcn.r09_positive_pair_preparation.v1',registered_at_unix=time.time(),
        state='metadata_preparation_only',dimension='data',
        hypothesis='Compare aligned real positive pair batching with unpaired batching using exactly the same target examples, condition masks, optimizer and decoding. D06 already establishes fixed-pair memorisation; test held-out response and editing utility.',
        input_sha256={p:digest(ROOT/p) for p in sources},
        train_source=str(support_path.relative_to(ROOT)),train_pairs=train,
        train_rows=[by_id[c] for c in sorted(train_ids)],
        train_canonical_sha256={c:support['canonical_sha256'][c] for c in sorted(train_ids)},
        train_canonical_root='inputs/d09_real_pair_support/canon',
        validation_rows=validation,validation_canonical_root='inputs/r09_positive_pairs/canon',
        validation_selection='One previously unused validation audio per mode x six density buckets, fixed hash ordering; report missing buckets. Canonical validation and real metric baseline pending. No source reselection after model outputs.',
        missing_validation_cells=missing,excluded_prior_audio=sorted(excluded_audio),
        condition_views='All applicable conditions plus each changed single field. Identical views and frequencies in both batching arms; no negative-target training, no new loss or architecture.',
        validation_controls_from='Fixed supported bands independent of hidden validation targets; exact panel must freeze before inference.',
        local_execution_intent=dict(device='cpu',threads=2,rss_gib=8,formal_steps_per_arm=1000,
            maximum_total_wall_s=10800,resource_feasibility_required=True,optimizer_probe_required=True,
            formal_training_registered=False),
        canonical_transfer=dict(maximum_bytes=32*2**20,per_file_maximum_bytes=4*2**20,timeout_s=120,concurrency=2),
        remote_compute_started=False,model_promotion=False,
        limitations=['Small training cohort; fixed-batch memory is not held-out or whole-song utility.',
            'Single-field band differences can co-vary in unrequested attributes.',
            'Metadata density is not a calibrated difficulty rating.',
            'Previously diagnosed low-density Phira remains a regression panel, not fresh validation.'])
    out = ROOT/'bench/rhythm-v2/r09-positive-pairs-prepare-v1.json'
    if out.exists(): raise FileExistsError('preparation plan already frozen')
    out.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(train_pairs=len(train),train_audio=len(train_audio),train_charts=len(train_ids),
        validation_audio=len(validation),missing=missing),ensure_ascii=False))


if __name__=='__main__': main()
