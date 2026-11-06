"""Freeze benchmark v2 protocols from a request tape (metadata only; no model, no GPU).

Writes two JSON files:
  <out>/teacher-panel.json     audio-dependence-panel-v1 (~200 val windows, both modes)
  <out>/generation-panel.json  benchmark-v2-generation-panel (free + silence + anchors)
and prints their SHA-256 for plan['benchmark_v2'].

  .venv/bin/python -m scripts.freeze_benchmark_v2 --tape inputs/native-v2-d94-580/tape.json \
      --out reports/benchmark-v2/native-v2-d94-580
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.checkpoint_resume import file_sha

ROOT = Path(__file__).resolve().parents[1]


class TapePanel:
    """Records/charts/targets of a request tape; ``features`` is attached by callers."""

    def __init__(self, root, tape, *, features=None, audios=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root/tape).read_text())
        if not self.manifest['complete'] or self.manifest['failures']:
            raise ValueError('Request tape is incomplete or failed')
        charts = self.manifest['charts']
        if audios is not None:
            charts = [c for c in charts if c['audio_sha256'] in audios]
        self.charts, self.records, self.features = charts, [], features
        for chart_index, chart in enumerate(charts):
            for request in chart['requests']:
                self.records.append(dict(request, chart_index=chart_index, chart_id=chart['chart_id'],
                    audio_sha256=chart['audio_sha256'], component=chart['component'], split=chart['split'],
                    mode=chart['mode'], stratum=chart['stratum']))
        self._arrays = lru_cache(maxsize=8)(self._read)

    def _read(self, chart_index):
        with np.load(self.root/self.charts[chart_index]['target_path'], allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in archive.files}

    def targets(self, index):
        row = self.records[index]
        values = self._arrays(row['chart_index'])
        sl = slice(row['target_start'], row['target_stop'])
        return [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h, t, k, l in
                zip(*(values[key][sl] for key in ('onset_us', 'end_us', 'kind', 'lane')))]


def freeze(root, tape, *, windows=200, per_mode_positive=6, per_mode_silence=2, anchors_per_mode=2,
           seed=20261101, audios=None):
    from beatfcn.train.audio_dependence import freeze_teacher_panel
    from beatfcn.train.learning_evaluation import freeze_generation_panel
    panel = TapePanel(root, tape, audios=audios)
    tape_sha = file_sha(Path(root)/tape)
    teacher = freeze_teacher_panel(panel, windows=windows, seed=seed, tape_sha256=tape_sha)
    generation = freeze_generation_panel(panel, per_mode_positive=per_mode_positive,
        per_mode_silence=per_mode_silence, anchors_per_mode=anchors_per_mode, seed=seed, tape_sha256=tape_sha)
    for protocol in (teacher, generation):
        protocol['tape'] = str(tape)
        protocol['index_space'] = ('full tape' if audios is None else
                                   'tape filtered to locally available audios (smoke only)')
    return teacher, generation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tape', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--windows', type=int, default=200)
    p.add_argument('--per-mode-positive', type=int, default=6)
    p.add_argument('--per-mode-silence', type=int, default=2)
    p.add_argument('--anchors-per-mode', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261101)
    a = p.parse_args()
    teacher, generation = freeze(a.root, a.tape, windows=a.windows, per_mode_positive=a.per_mode_positive,
        per_mode_silence=a.per_mode_silence, anchors_per_mode=a.anchors_per_mode, seed=a.seed)
    out = a.out if a.out.is_absolute() else a.root/a.out
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, value in (('teacher-panel.json', teacher), ('generation-panel.json', generation)):
        path = out/name
        if path.exists():
            raise FileExistsError(f'{path} exists; frozen protocols are never overwritten')
        path.write_text(json.dumps(value, indent=1)+'\n')
        result[name] = dict(path=str(path.relative_to(a.root) if path.is_relative_to(a.root) else path),
                            sha256=file_sha(path))
    result['teacher_windows'] = len(teacher['windows'])
    result['generation_samples'] = len(generation['samples'])
    result['anchors'] = len(generation['anchors'])
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
