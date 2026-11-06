"""Conditional full-campaign costs; never a launcher or a measured GPU quote."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'invalid {name}')
    return value


def encoded_seconds(duration, *, context=360, window=300, overlap=100):
    positive(duration, 'duration')
    if not 0 <= overlap < window <= context:
        raise ValueError('invalid long-song window')
    if duration <= context:
        return duration
    extra_windows = math.ceil((duration - window) / (window - overlap))
    return duration + extra_windows * overlap


def estimate(scenario, platform, audio_seconds, *, storage_gb, retention_hours,
             egress_gib, reserve_fraction, remaining_usd):
    """s1_seconds is per effective update, including all gradient accumulation."""
    for name, value in scenario.items():
        positive(value, name, zero=name.endswith('hours'))
    positive(audio_seconds, 'audio_seconds')
    for name, value in [('storage_gb', storage_gb), ('retention_hours', retention_hours),
                        ('egress_gib', egress_gib), ('reserve_fraction', reserve_fraction),
                        ('remaining_usd', remaining_usd)]:
        positive(value, name, zero=True)
    rate = positive(platform['compute_usd_h'], 'compute_usd_h')
    phases = {
        'layer_and_health_probe': scenario['probe_hours'],
        'full_corpus_encode_including_io': audio_seconds / scenario['encode_audio_seconds_per_wall_second'] / 3600,
        's1_10000_effective_updates': 10000 * scenario['s1_seconds_per_effective_update'] / 3600,
        'ten_thousand_step_reviews': 10 * scenario['routine_review_minutes'] / 60,
        'three_expanded_reviews': 3 * scenario['expanded_review_minutes'] / 60,
        'best_checkpoint_audit': scenario['final_audit_minutes'] / 60,
        's2_1000_decoder_lora_updates': 1000 * scenario['s2_seconds_per_effective_update'] / 3600,
        'four_s2_reviews': 4 * scenario['s2_review_minutes'] / 60,
        'setup_compile_save_and_transfer_wait': scenario['overhead_hours'],
    }
    compute_hours = sum(phases.values())
    if compute_hours > retention_hours:
        raise ValueError('retention shorter than scenario compute time')
    if platform['storage_kind'] == 'runpod_network':
        storage = storage_gb * .07 * math.ceil(retention_hours) / 720
    elif platform['storage_kind'] == 'runpod_local_volume':
        # Keep data between active phases; stopping the GPU does not stop this disk bill.
        storage = storage_gb * (.1 * compute_hours + .2 * (retention_hours - compute_hours)) / 720
    elif platform['storage_kind'] == 'modal_no_free_allowance':
        storage = storage_gb * 1e9 / 2**30 * .09 * retention_hours / 720
    else:
        raise ValueError('unknown storage type')
    migration = egress_gib * .04 if platform['provider'] == 'runpod' else 0.
    compute = compute_hours * rate
    subtotal = compute + storage + migration
    reserve = subtotal * reserve_fraction
    total = subtotal + reserve
    return dict(phase_hours=phases, compute_hours=compute_hours,
                phase_compute_usd={k: v * rate for k, v in phases.items()},
                compute_usd=compute, storage_usd=storage, migration_egress_usd=migration,
                subtotal_usd=subtotal, contingency_usd=reserve,
                total_scenario_usd=total, extra_above_remaining_usd=max(0., total-remaining_usd),
                fits_numeric_scenario=total <= remaining_usd,
                dispatch_ready=False, actual_performance_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    inventory = json.loads(Path(config['inventory']).read_text())
    budget = json.loads(Path(config['budget']).read_text())
    remaining = budget['unallocated_margin_usd']
    if abs(budget['approved_total_usd'] - budget['campaign_committed_upper_usd'] - remaining) > 1e-7:
        raise ValueError('inconsistent budget ledger')
    cohort = inventory['cohorts'][config['cohort']]
    duration = sum(encoded_seconds(x) for x in cohort['audio_duration_by_sha256'].values())
    results = []
    for name, scenario in config['scenarios'].items():
        for platform_name, platform in config['platforms'].items():
            for egress in config['egress_gib_scenarios'] if platform['provider'] == 'runpod' else [0]:
                result = estimate(scenario, platform, duration,
                    storage_gb=config['storage_gb'], retention_hours=config['retention_hours'],
                    egress_gib=egress, reserve_fraction=config['reserve_fraction'], remaining_usd=remaining)
                results.append(dict(scenario=name, platform=platform_name, egress_gib_assumed=egress, **result))
    paths = [str(args.config), config['inventory'], config['budget'], config['pricing'], __file__]
    report = dict(schema='beatfcn.mert2_campaign_cost.v1', date='2026-10-25',
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        source_audio_hours=cohort['audio_hours'], encoded_audio_hours_with_long_song_overlap=duration/3600,
        corpus_scope=config['cohort'], missing_audio_rows=cohort['missing_or_invalid_audio_rows'],
        approved_total_usd=budget['approved_total_usd'], committed_upper_usd=budget['campaign_committed_upper_usd'],
        remaining_usd=remaining, storage_gb=config['storage_gb'], retention_hours=config['retention_hours'],
        reserve_fraction=config['reserve_fraction'], results=results,
        quote_is_sensitivity_analysis=True, dispatch_ready=False, new_cloud_spend_usd=0,
        limitations=config['limitations'])
    if args.output.exists():
        raise FileExistsError('preserve the prior scenario report')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), cases=len(results), remaining_usd=remaining,
                         source_hours=cohort['audio_hours'], encoded_hours=duration/3600)))


if __name__ == '__main__':
    main()
