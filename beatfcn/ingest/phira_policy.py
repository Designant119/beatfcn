"""Phira strata and auditable metadata screening; neither proves native playability."""
import hashlib
import json
import re

from .coverage import phira_native_difficulty


STRATA_VERSION = 'phira-tag-level-v2'
SCREEN_VERSION = 'phira-native-metadata-v1'
STANDARD_TAGS = frozenset({'EZ', 'HD', 'IN', 'AT'})
LEVEL_BANDS = ((0, 4, '0-4'), (4, 8, '4-8'), (8, 11, '8-11'),
               (11, 13, '11-13'), (13, 15, '13-15'), (15, 16, '15-16'),
               (16, None, '16+'))
EXCLUDED_TAGS = frozenset({'visual', 'troll', 'sp', '4k', '5k', '6k', '7k', '8k',
    '9k', '10k', 'mania', 'osu!mania', 'adofai', '冰与火之舞', '转谱', '转换谱',
    '表演谱', '演出谱', '观赏谱', '绘图谱', '自动播放'})
REVIEW_PATTERNS = (
    ('possible_gameplay_simulation', re.compile(r'(?<![A-Za-z0-9])(?:[4-9]|10)\s*K(?![A-Za-z0-9])|ADOFAI|冰与火之舞|osu!?\s*mania', re.I)),
    ('possible_conversion', re.compile(r'转谱|转换谱|转制谱|移植谱|(?:转换|转制|移植)自|MalodyVersusPhigros|\bM2P\b', re.I)),
    ('possible_performance_chart', re.compile(r'(?:纯)?(?:表演|演出|观赏|绘图|画图)谱|纯(?:表演|演出|观赏)|仅供观赏|无需操作|自动播放谱|以.{0,12}(?:表演|演出|绘图|画图)为主', re.I)),
    ('possible_other_editor_origin', re.compile(r'使用\s*Malody|Malody\s*制作', re.I)),
    ('possible_special_interaction', re.compile(r'(?:点击|击打|按压).{0,5}任意(?:区域|位置)|模拟.{0,12}(?:不是|并非).{0,8}(?:击打|判定)')),
)
NEGATED = re.compile(r'(?:并非|不是|没有|不含|不包含|不属于|非|无)\s*$')


def native_tag(row):
    label = row.get('level') or row.get('difficulty_label') or ''
    return re.split(r'[\s.]', str(label).strip().upper())[0]


def phira_stratum(row):
    tag = native_tag(row)
    value, evidence = phira_native_difficulty(row)
    band = next((name for lower, upper, name in LEVEL_BANDS
                 if value is not None and value > 0 and value >= lower
                 and (upper is None or value < upper)), None)
    return dict(version=STRATA_VERSION, tag=tag, native_difficulty=value,
                numeric_band=band, evidence=evidence,
                key=f'{tag}/{band}' if tag in STANDARD_TAGS and band else None,
                calibrated_difficulty=False)


def metadata_identity(row):
    """Bind a manual decision to exactly the fields used by this screen."""
    fields = dict(native_id=str(row.get('id', row.get('native_id', ''))),
                  level=row.get('level') or row.get('difficulty_label') or '',
                  title=row.get('name') or row.get('title') or '',
                  description=row.get('description'), tags=row.get('tags'))
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def screen_metadata(row):
    """Exclude explicit nonstandard labels/tags; ambiguous prose needs review.

    A clean result is only absence of these metadata signals. It does not certify
    the packaged chart, its animation/visibility, audio clock, or authorship.
    """
    tags = row.get('tags') or []
    if isinstance(tags, str):
        tags = tags.replace(',', ' ').split()
    tags = {str(t).strip().lower() for t in tags}
    tag = native_tag(row)
    evidence = []
    if tag not in STANDARD_TAGS:
        evidence.append(dict(reason='nonstandard_level_tag', field='level', value=tag))
    for value in sorted(tags & EXCLUDED_TAGS):
        evidence.append(dict(reason='nonstandard_partition_or_gameplay_tag', field='tags', value=value))
    if evidence:
        status = 'excluded'
    else:
        for field in ('name', 'title', 'description', 'tags'):
            text = ' '.join(sorted(tags)) if field == 'tags' else str(row.get(field) or '')
            for reason, pattern in REVIEW_PATTERNS:
                for match in pattern.finditer(text):
                    if NEGATED.search(text[max(0, match.start() - 8):match.start()]):
                        continue
                    evidence.append(dict(reason=reason, field=field,
                        value=match.group(), excerpt=text[max(0, match.start()-24):match.end()+36]))
        status = 'review_required' if evidence else 'eligible_metadata_only'
    result = dict(version=SCREEN_VERSION, status=status, evidence=evidence,
                  public_context_present='tags' in row and 'description' in row,
                  native_content_verified=False)
    review = row.get('phira_metadata_review')
    if review:
        valid = (review.get('screen_version') == SCREEN_VERSION
                 and review.get('metadata_identity') == metadata_identity(row)
                 and review.get('decision') in {'excluded', 'review_required', 'eligible_metadata_only'}
                 and bool(review.get('reason')))
        # A stale annotation must never silently restore eligibility. Explicit
        # nonstandard tags also cannot be cleared by a prose-only review.
        if not valid:
            result.update(status='review_required', review_error='stale_or_invalid_metadata_review')
        elif status != 'excluded' or review['decision'] == 'excluded':
            result.update(status=review['decision'], manual_review=review,
                          automated_status=status)
    return result
