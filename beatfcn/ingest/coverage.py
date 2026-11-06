"""Difficulty evidence and conservative per-tier training-scale accounting."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import re


def policy_identity(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def native_tier(mode, value, policy):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    if value < 0 or (mode == 'phira' and value == 0):
        return None
    for tier in policy['tiers']:
        lower, upper = policy['native_boundaries'][mode][tier]
        if value >= lower and (upper is None or value < upper):
            return tier
    return None


def phira_native_difficulty(row):
    """Use numeric self-report, check integer level, retain uncertain labels."""
    values = []
    try:
        number = float(row.get('difficulty'))
        if math.isfinite(number) and number > 0:
            values.append(number)
    except (ValueError, TypeError):
        pass
    label = row.get('level') or row.get('difficulty_label') or ''
    match = re.search(r'(?:LV\.?\s*|^(?:EZ|HD|IN|AT)[\s.]+)(\d+(?:\.\d+)?)', label.upper())
    if match and float(match[1]) > 0:
        values.append(float(match[1]))
    if not values:
        return None, 'missing_numeric_difficulty'
    if max(values) - min(values) > 1:
        return None, 'conflicting_numeric_difficulty'
    return values[0], 'self_reported_not_calibrated'


def quota_report(rows, policy):
    """Only count unique admitted, feature-verified train targets.

    Rows must already carry native difficulty evidence under this policy. Raw
    manifest counts are reported separately and never inferred to be ready.
    """
    cells = {f'{mode}/{tier}': dict(required=minimum, raw_candidates=0,
              technical_unique_train=0, feature_ready_unique_train=0,
              accepted_unique_train=0, unique_audio=0, unique_song_groups=0)
             for mode, minimum in policy['minimum_unique_train_charts_per_tier'].items()
             for tier in policy['tiers']}
    groups = defaultdict(list)
    exclusions = Counter()
    held_out_audio = {r.get('audio_sha256') for r in rows if r.get('split') in ('val', 'test')}
    held_out_groups = {r.get('song_group') for r in rows if r.get('split') in ('val', 'test')}
    expected_policy = policy_identity(policy)
    for row in rows:
        mode, tier = row.get('mode'), row.get('difficulty_tier')
        key = f'{mode}/{tier}'
        if key not in cells:
            exclusions['unknown_difficulty'] += 1
            continue
        cells[key]['raw_candidates'] += 1
        if row.get('split') != 'train':
            exclusions['outside_train'] += 1
            continue
        if row.get('audio_sha256') in held_out_audio or row.get('song_group') in held_out_groups:
            exclusions['held_out_song_overlap'] += 1
            continue
        if row.get('difficulty_policy_sha256') != expected_policy:
            exclusions['difficulty_policy_not_verified'] += 1
            continue
        audio, target = row.get('audio_sha256'), row.get('pure_rhythm_hash')
        if not audio or not target:
            exclusions['missing_content_identity'] += 1
            continue
        groups[(mode, audio, target)].append(row)
    seen_audio, seen_song = defaultdict(set), defaultdict(set)
    for duplicates in groups.values():
        tiers = {r['difficulty_tier'] for r in duplicates}
        if len(tiers) != 1:
            exclusions['duplicate_targets_with_conflicting_tiers'] += len(duplicates)
            continue
        exclusions['duplicate_uploads'] += len(duplicates) - 1
        key = f'{duplicates[0]["mode"]}/{duplicates[0]["difficulty_tier"]}'
        valid = [r for r in duplicates if r.get('technical_audit_pass') is True
                 and not r.get('quarantined') and not r.get('split_conflict')
                 and r.get('difficulty_review_status') == 'verified']
        if valid:
            cells[key]['technical_unique_train'] += 1
        feature_ready = [r for r in valid if r.get('feature_cache_status') == 'verified']
        if feature_ready:
            cells[key]['feature_ready_unique_train'] += 1
        accepted = [r for r in feature_ready if r.get('training_ready') is True and r.get('song_group')]
        if accepted:
            cells[key]['accepted_unique_train'] += 1
            seen_audio[key].add(accepted[0]['audio_sha256'])
            seen_song[key].add(accepted[0]['song_group'])
    for key, cell in cells.items():
        cell.update(unique_audio=len(seen_audio[key]), unique_song_groups=len(seen_song[key]),
                    shortfall=max(0, cell['required'] - cell['accepted_unique_train']))
    return dict(cells=cells, exclusions=dict(exclusions),
                scale_gate_passed=all(c['shortfall'] == 0 for c in cells.values()),
                scope='Verified target identities, difficulty, technical admission and features; metadata supply is separate.')
