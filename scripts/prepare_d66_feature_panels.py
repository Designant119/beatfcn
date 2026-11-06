"""Freeze nested source-screened panels and bounded, read-only cache transfers."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from beatfcn.train.local_cohort import stratified_song_order

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    names = ['reports/d66_source_gates_v1.json', 'reports/d66_source_inventory_v2.json',
        'reports/d66_added_feature_inventory_v1.json', 'reports/d66_scale_transfer_v1.json',
        'reports/r09_feature_transfer.json', 'inputs/d66_local_cohort/evaluation20-candidates-v1.json']
    source, old, added, scale, previous, evaluation = [json.loads((ROOT/n).read_text()) for n in names]
    if not all(r['complete'] for r in (old, added, scale, previous)) or scale['failures']:
        raise ValueError('Complete source inventories and scale proof required')
    entries = defaultdict(dict)
    for r in old['records']:
        if r['kind'] == 'taps' and r['available']:
            for f in r['files']:
                entries[r['audio_sha256']][Path(f['path']).name] = f
    for r in added['records']:
        if r['kind'] == 'pupu_tap':
            entries[r['audio_sha256']][Path(r['path']).name] = r
    needed_names = {'L9.npy', 'L16.npy', 'L24.npy'}
    eligible = [r for r in source['eligible'] if set(entries[r['audio_sha256']]) == needed_names]
    unavailable = [r['chart_id'] for r in source['eligible'] if set(entries[r['audio_sha256']]) != needed_names]
    train = {mode: stratified_song_order([r for r in eligible if r['split'] == 'train' and r['mode'] == mode],
        seed=20261025) for mode in ('phira', 'mania_4k')}
    largest = min(len(train['phira'])//3, len(train['mania_4k']))*4
    if largest < 100:
        raise ValueError('Insufficient independent songs for initial panel')
    tiers = {str(n): [r['component'] for r in train['phira'][:n*3//4]+train['mania_4k'][:n//4]]
        for n in (100, largest)}
    val = [r for r in eligible if r['split'] == 'val']
    validation = sorted({r['component'] for r in val})
    if set(validation) & set(tiers[str(largest)]):
        raise ValueError('Nested panels leak songs across splits')
    if not {r['chart_id'] for r in evaluation['records']} <= {r['chart_id'] for r in val}:
        raise ValueError('Previously fixed 20-song panel lost feature availability')
    # Equal targets within a connected song are one supervision item. Different
    # targets and audio revisions remain, but never count as independent songs.
    distinct = {}
    for r in eligible:
        if r['component'] in set(tiers[str(largest)]) | set(validation):
            distinct.setdefault((r['component'], r['mode'], r['model_rhythm_sha256']), r)
    charts = list(distinct.values())
    prior = {r['path']: r for r in previous['files']}
    songs = []
    components = set(tiers['100']) | set(validation)
    initial_audio = {r['audio_sha256'] for r in charts if r['component'] in components}
    initial_jobs, later_jobs, reused = [], [], []
    for audio in sorted({r['audio_sha256'] for r in charts}):
        paths = {}
        for name, remote in sorted(entries[audio].items()):
            path = remote['path'].lstrip('/')
            p = prior.get(path)
            local = ROOT/p['local'] if p else None
            if (p and local.is_file() and local.stat().st_size == remote['bytes']
                    and sha(local) == p['sha256']):
                paths[name] = p['local']; reused.append(p)
            else:
                paths[name] = 'inputs/d66_local_cohort/features/'+path
                job = dict(path=path, bytes=remote['bytes'], audio_sha256=audio)
                (initial_jobs if audio in initial_audio else later_jobs).append(job)
        songs.append(dict(audio_sha256=audio, paths=paths,
            feature_bytes=sum(e['bytes'] for e in entries[audio].values()),
            initial_panel=audio in initial_audio))
    output = ROOT/'inputs/d66_local_cohort/feature-panels-v1.json'
    report = ROOT/'reports/d66_feature_preparation_v1.json'
    if output.exists() or report.exists():
        raise FileExistsError('Nested feature panels already frozen')
    registered = dict(id='D66-feature-panels-v1', training_ready=False, seed=20261025,
        source_sha256={n: sha(ROOT/n) for n in names},
        code_sha256={n: sha(ROOT/n) for n in ['scripts/prepare_d66_feature_panels.py','beatfcn/train/local_cohort.py']},
        tiers=tiers, validation=validation, charts=charts, songs=songs,
        scales=scale['files'][0]['local'], scales_sha256=scale['files'][0]['sha256'],
        reused_files=reused, missing_cache_charts=unavailable,
        actual_training_updates=0, evaluation20=names[-1],
        limits='Availability and source screen only; array, clock and loader gates remain.')
    first_bytes = sum(r['bytes'] for r in initial_jobs)
    later_bytes = sum(r['bytes'] for r in later_jobs)
    if shutil.disk_usage(ROOT).free < first_bytes+later_bytes+32*2**30:
        raise ValueError('Full nested cache would cross the local storage reserve')
    output.write_text(json.dumps(registered, ensure_ascii=False, indent=2)+'\n')
    plans = []
    for phase, jobs in [('initial', initial_jobs), ('expansion', later_jobs)]:
        path = ROOT/f'bench/rhythm-v2/d66-features-{phase}-v1.json'
        if path.exists():
            raise FileExistsError(path)
        size = sum(r['bytes'] for r in jobs)
        plan = dict(id='D66-features-'+phase+'-v1', files=jobs,
            local_root='inputs/d66_local_cohort/features',
            report=f'reports/d66_features_{phase}_transfer_v1.json',
            concurrency=4, timeout_s=900, maximum_transfer_bytes=size+2*2**30,
            minimum_free_bytes=32*2**30, feature_panel_sha256=sha(output))
        path.write_text(json.dumps(plan,indent=2)+'\n'); plans.append(str(path.relative_to(ROOT)))
    summary = dict(id='D66-feature-preparation-v1', training_ready=False,
        feature_panels=str(output.relative_to(ROOT)), panel_sha256=sha(output),
        tiers={n: len(v) for n,v in tiers.items()}, validation_songs=len(validation),
        initial_audio=len(initial_audio), unique_audio=len(songs), charts=len(charts),
        initial_download_bytes=first_bytes, later_download_bytes=later_bytes,
        reused_bytes=sum(r['bytes'] for r in reused), missing_cache_charts=unavailable,
        download_plans=plans, new_cloud_compute_calls=0, optimizer_updates=0,
        chart_distribution=dict(Counter(r['split']+'/'+r['mode']+'/'+r['stratum'] for r in charts)))
    report.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='chart_distribution'}))


if __name__ == '__main__':
    main()
