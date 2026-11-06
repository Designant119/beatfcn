"""Export raw completion evidence using the repository's existing timeline view."""
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
    report = json.loads((ROOT / 'reports/completion_quality_v1.json').read_text())
    if not report['complete']:
        raise ValueError('Review only terminal evidence')
    sources = {r['chart_id']: r for r in json.loads((ROOT / 'reports/completion_audio_sources_v1.json').read_text())['records']}
    features = {r['audio_sha256']: r for r in json.loads((ROOT / 'reports/d94_merged_feature_admission_v1.json').read_text())['records']}
    clock = {r['audio_sha256']: r for r in json.loads((ROOT / 'reports/d66_audio_clock_v1.json').read_text())['audio']}
    dest = ROOT / 'artifacts/completion-review-v2'
    dest.mkdir(exist_ok=False)
    links, manifest = [], []
    for row in report['songs']:
        chart = row['chart_id']
        source = sources[chart]
        raw = Path(source['audio_path'])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != source['audio_sha256']:
            raise ValueError('Original audio changed')
        feature = features[source['audio_sha256']]
        mel = np.load(ROOT / feature['complete_mel'], mmap_mode='r')[:feature['valid_mel_frames']]
        # The last valid mel frame can be partial. This mirrors the established
        # review recipe; raw event/audio clocks and model scores are unchanged.
        flux = audio_features(mel, len(mel) * 10_000)['flux']
        folder = dest / chart.replace(':', '-')
        folder.mkdir()
        # Sniff the verified compressed/container header; do not transcode audio.
        header = raw.read_bytes()[:16]
        suffix = '.ogg' if header[:4] == b'OggS' else ('.flac' if header[:4] == b'fLaC' else ('.wav' if header[:4] == b'RIFF' else '.mp3'))
        shutil.copyfile(raw, folder / ('audio' + suffix))
        sections = []
        for arm, result in row['outputs'].items():
            path = ROOT / result['artifact']
            data = json.loads(path.read_text())
            name = arm + '.svg'
            write_svg(folder / name, f"{chart} {row['task']} {arm}: missing notes only",
                      data['prediction'], data['hidden'], source['duration_us'], flux)
            detail = arm + '-detail.svg'
            duration = source['duration_us']
            # Deterministic most active music core, rather than selecting by model output.
            active = max(range(0, max(1, duration - 8_000_000 + 1), 8_000_000),
                         key=lambda t: float(np.mean(flux[t // 10000:min(len(flux), (t + 8_000_000) // 10000)])))
            write_svg(folder / detail, f"{chart} {row['task']} {arm}: active detail",
                      data['prediction'], data['hidden'], duration, flux,
                      scope=(active, min(duration, active + 8_000_000)))
            csv_name = arm + '.csv'
            shutil.copyfile(path.with_suffix('.csv'), folder / csv_name)
            sections.append(f'<h2>{html.escape(arm)}</h2><p>complete={result["complete"]}; hidden full-note F1={result["full_note_f1"]:.4f}; types={html.escape(str(result["predicted_types"]))}</p><a href="{csv_name}">visible / hidden / prediction CSV</a><img width="100%" src="{name}"><img width="100%" src="{detail}">')
            manifest.append(dict(chart_id=chart, arm=arm, raw_result=result['artifact'],
                                 raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 source_audio_sha256=source['audio_sha256'], svg=[str((folder / name).relative_to(ROOT)), str((folder / detail).relative_to(ROOT))]))
        (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px;background:#101720;color:#e2e8ef}a{color:#90caff}img{margin:12px 0}</style>'
            + f'<h1>{html.escape(chart)} · {html.escape(row["task"])}</h1><p>仅比较隐藏音符，保留音符没有加入F1。蓝/绿/橙/紫为tap/hold/flick/drag；空间未建模。图不是试听或试玩结论。</p><audio controls src="audio{suffix}"></audio>' + ''.join(sections))
        links.append(f'<li><a href="{folder.name}/index.html">{html.escape(chart)} · {html.escape(row["stratum"])} · {html.escape(row["task"])}</a></li>')
    (dest / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>参考补全原始结果</h1><p>终态：'
        + html.escape(report['status']) + f'；实际评价 {len(report["songs"])}/20 首。'
        + html.escape(str(report.get('stop_reason', ''))) + '</p><ul>' + ''.join(links) + '</ul>')
    (dest / 'manifest.json').write_text(json.dumps(dict(records=manifest, evaluated_songs=len(report['songs']),
        generated_only=True, actual_listening=False, native_playtest=False), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(review=str(dest.relative_to(ROOT)), songs=len(report['songs']))))


if __name__ == '__main__':
    main()
