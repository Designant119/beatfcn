"""Coverage-aware acquisition, separate from technical validation and training admission.

Ratings and votes are evidence about reception, not definitions of difficulty or
technical validity. Acquisition never makes a chart training-ready by itself.
"""
from collections import Counter, defaultdict
import hashlib
import math
import re

from beatfcn.events.group_split import work_key
from beatfcn.ingest.pipeline import assign_splits


STANDARD_TAGS = frozenset({'EZ', 'HD', 'IN', 'AT'})


def osu_metadata_fields(text):
    # Native .osu key/value lines commonly have a space after the colon.
    return {key: value.strip() for key, value in re.findall(
        r'^\s*(BeatmapID|BeatmapSetID|AudioFilename|Mode|CircleSize)\s*:(.*?)\s*$', text, re.M)}


def tag_of(row):
    return re.split(r'[\s.]', (row.get('level') or row.get('difficulty_label') or '').strip().upper())[0]


def phira_evidence(row):
    """Explicit signals; low confidence is not a negative quality label."""
    count = int(row.get('ratingCount') or row.get('rating_count') or 0)
    rating = row.get('rating')
    return {
        'reviewed': bool(row.get('reviewed')),
        'ranked': bool(row.get('ranked')),
        'stable': bool(row.get('stable')),
        'rating': rating, 'votes': count,
        'legacy_rating_gate': rating is not None and rating >= .90 and count >= 100,
        'confidence': 'unknown' if not count else 'limited' if count < 100 else 'many_votes',
        'musical_quality_verified': False,
    }


def phira_acquisition_eligible(row):
    tags = row.get('tags') or []
    if isinstance(tags, str):
        tags = tags.replace(',', ' ').split()
    return (bool(row.get('reviewed')) and bool(row.get('file'))
            and tag_of(row) in STANDARD_TAGS and not set(tags) & {'troll', 'visual'})


def declared_burden(row):
    """Conservative acquisition hint, never a measured difficulty or quality score.

    Both numeric fields are self-reported. Conflicting IN/AT Lv.15 and numeric 1
    must not manufacture a low-difficulty example. Events provide the real audit.
    """
    nums = []
    try:
        value = float(row.get('difficulty'))
        if math.isfinite(value) and value >= 0:
            nums.append(value)
    except (TypeError, ValueError):
        pass
    label = row.get('level') or row.get('difficulty_label') or ''
    match = re.search(r'(?:LV\.?\s*|^(?:EZ|HD|IN|AT)[\s.]+)(\d+(?:\.\d+)?)', label.upper())
    if match:
        nums.append(float(match[1]))
    value = max(nums) if nums else None
    return ('unknown' if value is None else '0-6' if value < 6 else '6-10' if value < 10 else '10+',
            len(nums) > 1 and max(nums) - min(nums) > 1)


def select_phira(rows, existing_ids, limit, *, per_song=4, seed=20261025):
    """Prioritize missing easy/medium coverage, retain same-song contrast charts.

    One pass round-robins cells and prefers unrepresented songs in each cell.
    Ranking/votes are retained as audit evidence, not a popularity sort. Existing
    harder charts do not consume the expansion's per-song allowance.
    """
    from beatfcn.train.data import selected_for_training

    existing = {str(x).removeprefix('phira:') for x in existing_ids}
    cells = defaultdict(list)
    for row in rows:
        if (str(row['id']) in existing or not phira_acquisition_eligible(row)
                or not selected_for_training(dict(row, source='phira',
                    difficulty_label=row.get('level') or row.get('difficulty_label')))):
            continue
        cells[(declared_burden(row)[0], tag_of(row))].append(row)
    order = [(burden, tag) for burden in ('0-6', '6-10', '10+', 'unknown')
             for tag in ('EZ', 'HD', 'IN', 'AT')]
    for bucket in cells.values():
        bucket.sort(key=lambda r: hashlib.sha256(f'{seed}:phira:{r["id"]}'.encode()).hexdigest())
    selected, counts, files = [], Counter(), set()
    while len(selected) < limit:
        progressed = False
        for cell in order:
            pool = [r for r in cells[cell] if counts[work_key(r.get('name'))] < per_song
                    and r['file'] not in files and work_key(r.get('name'))]
            if not pool:
                continue
            row = min(pool, key=lambda r: counts[work_key(r.get('name'))])
            cells[cell].remove(row)
            selected.append(row)
            counts[work_key(row['name'])] += 1
            files.add(row['file'])
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def incremental_splits(existing, additions):
    """Anchor new songs to frozen splits; quarantine bridges across old splits.

    Re-finalizing the entire corpus could move old validation songs into train.
    Existing rows are never changed, including their group IDs and clock recipe.
    """
    old_ids = {r['chart_id'] for r in existing}
    if old_ids & {r['chart_id'] for r in additions}:
        raise ValueError('additions must not replace existing chart identities')
    groups, chart_group, group_split = assign_splits(existing + additions)
    anchors = defaultdict(list)
    for row in existing:
        if row.get('split') in ('train', 'val', 'test'):
            anchors[chart_group[row['chart_id']]].append(row)
    result = []
    for row in additions:
        group = chart_group[row['chart_id']]
        anchor_rows = anchors[group]
        splits = {r['split'] for r in anchor_rows}
        out = dict(row)
        out['split_evidence'] = {
            'method': 'frozen old splits; union of exact audio and normalized work/title',
            'existing_chart_ids': sorted(r['chart_id'] for r in anchor_rows),
            'conflicting_splits': sorted(splits) if len(splits) > 1 else [],
        }
        if len(splits) > 1:
            out.update(split=None, song_group=group, split_conflict=True)
        elif anchor_rows:
            anchor = min(anchor_rows, key=lambda r: r['chart_id'])
            out.update(split=anchor['split'], song_group=anchor['song_group'], split_conflict=False)
        else:
            out.update(split=group_split[group], song_group=group, split_conflict=False)
        result.append(out)
    return result
