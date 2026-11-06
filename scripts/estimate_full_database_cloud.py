"""Sensitivity-only cost estimate for exact cloud full-database preparation."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
RATES = ROOT / 'reports/modal-120m-20261030/current-rates.json'
DEST = ROOT / 'reports/full-database-cloud-v1/encode-cost-scenarios-v1.json'
SCENARIOS = {'fast_assumption': 100., 'middle_assumption': 50., 'slow_assumption': 20.}
GPUS = {
    'L40S': .000542,
    'A100-80GB': .000694,
    'H100-SXM5': .001097,
}
CPU_CORES = 8
MEMORY_GIB = 32
STARTUP_S = 300
STOP_S = 180
# Empirical 2026-10-25 payload slope for two float16 MERT layers plus clocks:
# 262.1418396911787 GB / 568.8842007187037 audio hours.
PAYLOAD_GB_PER_AUDIO_HOUR = 262.1418396911787 / 568.8842007187037


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded_seconds(duration, context=360, window=300, overlap=100):
    if duration <= context:
        return duration
    return duration + math.ceil((duration - window) / (window - overlap)) * overlap


def scopes(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['audio_sha256']].append(row)
    result = {}
    for name, predicate in (
        ('parsed', lambda values: any(row.get('status') == 'parsed' for row in values)),
        ('kept', lambda values: any(row.get('status') == 'parsed' and row.get('kept') for row in values))):
        selected = []
        for values in groups.values():
            if not predicate(values):
                continue
            duration = next((row['audio_sec'] for row in values if row.get('audio_sec') is not None), None)
            if duration is None:
                raise ValueError(f"Scope {name} lacks audio duration for {values[0]['chart_id']}")
            selected.append(float(duration))
        result[name] = selected
    return result


def main():
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    rates = json.loads(RATES.read_text())
    if float(rates['h100_sxm5_usd_s']) != GPUS['H100-SXM5'] or float(rates['cpu_physical_core_usd_s']) != .0000131:
        raise ValueError('Public rate source changed; refresh pricing before using this report')
    result = dict(schema='beatfcn.full-database-cloud-cost.v1', dispatch_ready=False,
        cloud_compute_calls=0, paid_actions=0, manifest=str(MANIFEST.relative_to(ROOT)),
        manifest_sha256=digest(MANIFEST), rates_source=str(RATES.relative_to(ROOT)),
        rates_sha256=digest(RATES), assumptions=dict(
            encode_audio_seconds_per_wall_second=SCENARIOS,
            cpu_cores=CPU_CORES, memory_gib=MEMORY_GIB,
            startup_s=STARTUP_S, stop_s=STOP_S,
            mert_layers=(12, 23), output_dtype='float16',
            long_song_context_s=360, long_window_s=300, overlap_s=100,
            payload_gb_per_audio_hour=PAYLOAD_GB_PER_AUDIO_HOUR,
            storage_note='Existing raw+canon+mel inventory is approximately 150GiB; two-layer MERT remains below the 1TiB included Modal volume allowance by this estimate.'),
        limitations=['Throughput values are explicit sensitivity assumptions, not measurements.',
            'GPU model choice does not imply equal MERT speed; a paid bounded pilot must measure it.',
            'Public prices are not account invoices or authorization; rates must be rechecked before dispatch.',
            'Payload slope is scaled from the 2026-10-25 inventory, not from newly written full-database receipts.',
            'No target materialization, hash verification, failure retry, repair transfer or egress is included.',
            'This report does not authorize any paid cloud call.'])
    cpu_memory_rate = CPU_CORES * .0000131 + MEMORY_GIB * .00000222
    cases = []
    for scope, durations in scopes(rows).items():
        raw = sum(durations)
        encoded = sum(map(encoded_seconds, durations))
        payload = raw / 3600 * PAYLOAD_GB_PER_AUDIO_HOUR
        for scenario, speed in SCENARIOS.items():
            encode_s = encoded / speed + STARTUP_S + STOP_S
            for gpu, gpu_rate in GPUS.items():
                rate = gpu_rate + cpu_memory_rate
                cases.append(dict(scope=scope, scenario=scenario, gpu=gpu,
                    audio_count=len(durations), raw_audio_hours=raw / 3600,
                    encoded_audio_hours=encoded / 3600, estimated_payload_gb=payload,
                    encode_plus_startup_hours=encode_s / 3600,
                    rate_usd_s=rate, compute_upper_usd=encode_s * rate))
    result['scopes'] = {scope: dict(audio_count=len(values), raw_audio_hours=sum(values) / 3600,
        encoded_audio_hours=sum(map(encoded_seconds, values)) / 3600,
        estimated_payload_gb=sum(values) / 3600 * PAYLOAD_GB_PER_AUDIO_HOUR)
        for scope, values in scopes(rows).items()}
    result['cases'] = cases
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['scopes'], indent=2))
    for scope in ('parsed', 'kept'):
        print('\n', scope)
        for case in cases:
            if case['scope'] == scope:
                print(case['scenario'], case['gpu'], round(case['compute_upper_usd'], 3))


if __name__ == '__main__':
    main()
