"""Raw song/audio/timeline evidence for the terminal full-stack comparison."""
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np

from beatfcn.model.benchmark import audio_features
from scripts.review_local_song_rollouts import write_svg

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'reports/fullstack_mamba_quality_v2.json').read_text())
    if not report['complete']:
        raise ValueError('Only terminal, immutable quality evidence may be reviewed')
    sources = {r['chart_id']: r for r in json.loads((ROOT / 'reports/completion_audio_sources_v1.json').read_text())['records']}
    features = {r['audio_sha256']: r for r in json.loads((ROOT / 'reports/d94_merged_feature_admission_v1.json').read_text())['records']}
    destination = ROOT / 'artifacts/fullstack-mamba-v1/review'
    destination.mkdir(exist_ok=False)
    links, manifest = [], []
    for row in report['songs']:
        source = sources[row['chart_id']]
        raw = Path(source['audio_path'])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != source['audio_sha256']:
            raise ValueError('Original audio changed')
        f = features[source['audio_sha256']]
        mel = np.load(ROOT / f['complete_mel'], mmap_mode='r')[:f['valid_mel_frames']]
        flux = audio_features(mel, len(mel) * 10000)['flux']
        folder = destination / row['chart_id'].replace(':', '-')
        folder.mkdir()
        header = raw.read_bytes()[:16]
        extension = '.ogg' if header[:4] == b'OggS' else ('.flac' if header[:4] == b'fLaC' else ('.wav' if header[:4] == b'RIFF' else '.mp3'))
        shutil.copyfile(raw, folder / ('audio' + extension))
        duration = source['duration_us']
        active = max(range(0, max(1, duration - 8_000_000 + 1), 8_000_000),
                     key=lambda t: float(np.mean(flux[t // 10000:min(len(flux), (t + 8_000_000) // 10000)])))
        body = []
        for arm, output in row['outputs'].items():
            path = ROOT / output['artifact']
            data = json.loads(path.read_text())
            write_svg(folder / (arm + '.svg'), f"{row['chart_id']} {row['task']} {arm}: hidden notes only",
                      data['prediction'], data['hidden'], duration, flux)
            write_svg(folder / (arm + '-detail.svg'), f"{row['chart_id']} {row['task']} {arm}: audio-active detail",
                      data['prediction'], data['hidden'], duration, flux, scope=(active, min(duration, active + 8_000_000)))
            shutil.copyfile(path.with_suffix('.csv'), folder / (arm + '.csv'))
            body.append(f'<h2>{html.escape(arm)}</h2><p>complete={output["complete"]}; F1={output["full_note_f1"]:.4f}; types={html.escape(str(output["predicted_types"]))}</p><a href="{arm}.csv">visible / hidden / prediction CSV</a><img width="100%" src="{arm}.svg"><img width="100%" src="{arm}-detail.svg">')
            manifest.append(dict(chart_id=row['chart_id'], arm=arm, raw_result=output['artifact'],
                raw_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_audio_sha256=source['audio_sha256']))
        page = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px;background:#101720;color:#e2e8ef}a{color:#90caff}img{margin:12px 0}</style>'
        page += f'<h1>{html.escape(row["chart_id"])} · {html.escape(row["task"])}</h1><p>仅对比新增预测和隐藏目标；保留人谱未加入F1。未建模Phira空间，图不是试听或试玩结论。</p><audio controls src="audio{extension}"></audio>' + ''.join(body)
        (folder / 'index.html').write_text(page)
        links.append(f'<li><a href="{folder.name}/index.html">{html.escape(row["chart_id"])} · {html.escape(row["stratum"])} · {html.escape(row["task"])}</a></li>')
    (destination / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Full stack Mamba 2 与 Transformer 原始补全结果</h1>'
        + f'<p>{html.escape(report["status"])}；实际评价{len(report["songs"])}/20首；{html.escape(str(report.get("stop_reason", "")))}</p><ul>' + ''.join(links) + '</ul>')
    (destination / 'manifest.json').write_text(json.dumps(dict(records=manifest, observed_songs=len(report['songs']),
        actual_listening=False, native_playtest=False), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(review=str(destination.relative_to(ROOT)), songs=len(report['songs']))))


if __name__ == '__main__':
    main()
