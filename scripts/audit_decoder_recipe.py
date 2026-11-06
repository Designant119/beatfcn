"""Compare new local mel with old cached mel from identical compressed audio."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.ingest import pipeline as P
from beatfcn.ingest.offset import flux_envelope


def main():
    torch.set_num_threads(2)
    path = ROOT / 'bench/rhythm-v2/data-expansion-clock-v2.json'
    plan = json.loads(path.read_text())
    receipt = json.loads((ROOT / plan['report']).read_text())
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt['complete'] and receipt['plan_sha256'] == identity
    by_path = {r['path']: r for r in receipt['files']}
    pairs = json.loads((ROOT / 'inputs/data_expansion_v1/clock_common_audio.json').read_text())
    root = ROOT / 'inputs/data_expansion_v1/corpus'
    for row in plan['additional_existing_audio_examples']:
        with zipfile.ZipFile(ROOT / row['archive']) as archive:
            names = {n.lower().rsplit('/', 1)[-1]: n for n in archive.namelist()}
            info = yaml.safe_load(archive.read(names.get('info.yml') or names['info.yaml']).decode('utf-8-sig'))
            audio = archive.read(names[str(info['music']).lower().rsplit('/', 1)[-1]])
        assert hashlib.sha256(audio).hexdigest() == row['audio_sha256']
        with tempfile.TemporaryDirectory() as temp:
            result = P.ensure_audio(str(root), audio, None, temp)
        assert result['codec'] == row['codec']
        pairs.append(row)
    results = []
    for row in pairs:
        relative = f'mel10/{row["audio_sha256"]}.npy'
        previous = ROOT / plan['local_root'] / relative
        current = root / relative
        assert hashlib.sha256(previous.read_bytes()).hexdigest() == by_path[relative]['sha256']
        old = np.load(previous).astype(np.float32)
        new = np.load(current).astype(np.float32)
        assert np.isfinite(old).all() and np.isfinite(new).all()
        result = dict(chart_id=row['chart_id'], audio_sha256=row['audio_sha256'], codec=row['codec'],
                      old_shape=list(old.shape), new_shape=list(new.shape), passed=False,
                      current_mel_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
                      old_mel_sha256=by_path[relative]['sha256'])
        if old.shape == new.shape:
            a, b = flux_envelope(old), flux_envelope(new)
            correlations = {}
            for shift in range(-3, 4):
                x, y = (a[-shift:], b[:shift]) if shift < 0 else (a[:-shift], b[shift:]) if shift > 0 else (a, b)
                correlations[shift] = float(np.corrcoef(x, y)[0, 1])
            best = max(correlations, key=correlations.get)
            delta = old - new
            result.update(best_flux_shift_frames=best, flux_correlation_at_zero=correlations[0],
                          max_abs_delta=float(np.abs(delta).max()), mean_abs_delta=float(np.abs(delta).mean()),
                          passed=best == 0 and correlations[0] >= plan['comparison_gate']['minimum_flux_correlation'])
        results.append(result)
    output = dict(schema='beatfcn.decoder_recipe_parity.v1', complete=True, passed=all(r['passed'] for r in results),
                  plan_sha256=identity, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gate=plan['comparison_gate'], pairs=results,
                  limitation='Nine existing audios test mp3/vorbis timing consistency. This is not universal codec equivalence, audio/chart offset truth, or model improvement.',
                  model_training_started=False, new_cloud_compute=False)
    (ROOT / 'reports/data_expansion_decoder_parity.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=output['passed'], pairs=len(results),
                         minimum_correlation=min(r.get('flux_correlation_at_zero', 0) for r in results))))
    if not output['passed']:
        raise SystemExit('decoder parity gate failed; do not mix new features into training')


if __name__ == '__main__':
    main()
