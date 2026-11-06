"""Local, deterministic comparison. No cloud calls or paid resources.

Uses small synthetic test models; timings are not estimates of L40S throughput.
The time-history correction is present on BOTH sides of the performance comparison.
"""
import copy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

import numpy as np
import torch
from beatfcn.model.generation_batch import GenerationBatch
from beatfcn.model.p1 import P1Model
from beatfcn.model.sample import generate_core, core_context
from beatfcn.model import features as F
from beatfcn.events.requests import build_request
from events_helpers import synthetic_mania, synthetic_phira, DURATION
from test_generation_batch import fixture, metadata
from test_p1_model import CFG, song


def measured(run, repeats=5):
    run()  # warmup, outside measurements
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return {'seconds': samples, 'median_s': statistics.median(samples)}


def main():
    torch.set_num_threads(1)
    result = {'scope': 'local synthetic CPU fixtures; not a CUDA or real-checkpoint throughput claim',
              'history_fix_on_both_sides': True, 'metadata': {}, 'generation': {}}
    for ex, audio, groups in fixture():
        right = np.zeros((len(groups) + 1, 4), np.float32)
        values = {}
        for reuse in (False, True):
            def run():
                work = copy.copy(ex)
                work.groups, work.right = [], right[:1]
                if reuse:
                    cache = GenerationBatch(metadata(work, audio), work, 'cpu')
                    for i, group in enumerate(groups):
                        cache.append(group, right[i + 1])
                else:
                    for i in range(len(groups) + 1):
                        work.groups, work.right = groups[:i], right[:i + 1]
                        metadata(work, audio)
            values['cached' if reuse else 'rebuild'] = measured(run)
        values['speedup'] = values['rebuild']['median_s'] / values['cached']['median_s']
        values['prefix_groups'] = len(groups)
        result['metadata'][str(ex.mode)] = values
    for chart in (synthetic_mania(3), synthetic_phira(4)):
        request = build_request(chart, kind='from_scratch', seed=9, audio_id='a', audio_duration_us=DURATION)
        ex = F.build_core(request, chart.events, 8 * F.SEC, DURATION)
        holds, runs, heads = core_context(request, ex, F.line_index_of(chart.events))
        torch.manual_seed(5)
        model = P1Model(CFG).eval()
        with torch.no_grad():
            model.kern.weight.normal_(std=.2)
        audio = song(1)
        values, outputs = {}, []
        for reuse in (False, True):
            def run():
                return generate_core(model, ex, audio, holds, runs, heads, greedy=False, seed=4,
                                     reuse_metadata=reuse)
            groups = run()
            outputs.append([(g['t_rel_us'], g['notes']) for g in groups])
            values['cached' if reuse else 'rebuild'] = measured(run)
        if outputs[0] != outputs[1] or not outputs[0]:
            raise AssertionError('generation comparison must match and contain notes')
        values.update(groups=len(outputs[0]), output_equal=True,
                      speedup=values['rebuild']['median_s'] / values['cached']['median_s'])
        result['generation'][chart.mode] = values
    path = ROOT / 'reports/evaluation_metadata_timing_2026-10-25.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
