"""Freeze fine Phira strata and source exclusions without certifying native content."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.ingest.expansion import phira_acquisition_eligible
from beatfcn.ingest.phira_policy import (LEVEL_BANDS, SCREEN_VERSION, STRATA_VERSION,
                                       native_tag, phira_stratum, screen_metadata)
from beatfcn.train.data import (PHIRA_SELECTION_VERSION, PHIRA_MIN_RATING,
                               PHIRA_MIN_RATING_COUNT, phira_rating_metadata)

CATALOG = 'inputs/data_scale_v2/catalog_v1/phira/catalog.jsonl'
MANIFESTS = [
    'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl',
    'inputs/data_expansion_v1/corpus/manifest/expansion-candidates.jsonl',
    'inputs/data_scale_v2/phigros_resource_v1/admission_v3/manifest/candidates.jsonl',
    'inputs/d34_plain_packages/audit_v2/manifest/candidates.jsonl',
    'inputs/d56_phira_085/audit/manifest/candidates.jsonl',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit_record(row, review=None):
    meta = dict(row, source='phira', difficulty_label=row.get('level'))
    if review:
        meta['phira_metadata_review'] = review
    screen, stratum = screen_metadata(meta), phira_stratum(meta)
    reasons = []
    if not phira_acquisition_eligible(meta):
        reasons.append('original_acquisition_gate')
    rating, count = phira_rating_metadata(meta)
    rating_ok = rating is not None and rating >= PHIRA_MIN_RATING
    votes_ok = count is not None and count >= PHIRA_MIN_RATING_COUNT
    if not rating_ok:
        reasons.append('rating_gate')
    if not votes_ok:
        reasons.append('rating_vote_floor')
    if not screen['public_context_present']:
        reasons.append('missing_public_context')
    if screen['status'] != 'eligible_metadata_only':
        reasons.append(screen['status'])
    if stratum['key'] is None:
        reasons.append(stratum['evidence'] if stratum['tag'] in {'EZ', 'HD', 'IN', 'AT'}
                       else 'nonstandard_level_tag')
    return dict(native_id=str(meta['id']), title=meta.get('name'),
                metadata_row_sha256=hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                rating=rating, rating_count=count,
                rating_pass=rating_ok, votes_pass=votes_ok,
                original_acquisition_pass=phira_acquisition_eligible(meta),
                stratum=stratum, screen=screen, selection_reasons=reasons,
                metadata_candidate=not reasons, native_content_verified=False,
                training_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default='reports/d57_phira_strata_audit')
    args = parser.parse_args()
    output = ROOT / args.output_prefix
    paths = {suffix: output.with_suffix(suffix) for suffix in ('.json', '.md', '.rows.jsonl', '.quarantine.jsonl', '.corpus-overlay.jsonl')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Use a new output prefix; existing audits are immutable.')
    started = time.monotonic()
    review_path = ROOT / 'bench/rhythm-v2/d57-phira-metadata-reviews-v1.json'
    reviews = json.loads(review_path.read_text())
    if reviews['catalog_sha256'] != digest(ROOT / CATALOG):
        raise ValueError('Review catalog identity changed')
    catalog = lines(ROOT / CATALOG)
    if len({str(r['id']) for r in catalog}) != len(catalog):
        raise ValueError('Duplicate catalog IDs')
    rows = [audit_record(r, reviews['records'].get(str(r['id']))) for r in catalog]
    by_id = {r['native_id']: r for r in rows}
    base = [r for r in rows if r['rating_pass'] and r['votes_pass'] and r['original_acquisition_pass']]
    candidate = [r for r in rows if r['metadata_candidate']]
    matrices = {}
    for name, subset in [('all_catalog', rows), ('previous_metadata_candidates', base),
                         ('screened_metadata_candidates', candidate)]:
        matrix = defaultdict(Counter)
        for r in subset:
            s = r['stratum']
            matrix[s['tag'] if s['tag'] in {'EZ', 'HD', 'IN', 'AT'} else 'OTHER'][s['numeric_band'] or 'unknown'] += 1
        matrices[name] = {tag: dict(counts) for tag, counts in matrix.items()}
    overlay, seen = [], set()
    for filename in MANIFESTS:
        for row in lines(ROOT / filename):
            if row.get('source') != 'phira':
                continue
            if row['chart_id'] in seen:
                raise ValueError(f'Duplicate corpus ID: {row["chart_id"]}')
            seen.add(row['chart_id'])
            audit = by_id.get(str(row.get('native_id')))
            original = dict(row)
            # Retain the previous score/label decision as a separate value.
            try:
                prior_rating_label_pass = (.85 <= float(row.get('rating') or 0) <= 1
                                           and native_tag(row) in {'EZ', 'HD', 'IN', 'AT'})
            except (TypeError, ValueError):
                prior_rating_label_pass = False
            overlay.append(dict(chart_id=row['chart_id'], manifest=filename,
                source_row_sha256=hashlib.sha256(json.dumps(original, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                split=row.get('split'), prior_rating_label_pass=prior_rating_label_pass,
                current_catalog_audit=audit,
                metadata_candidate=bool(audit and audit['metadata_candidate']),
                metadata_to_package_identity_verified=False, training_ready=False,
                join_limitation='Native ID joins metadata only; package revision/content still requires verification.'))
    summary = dict(catalog_ids=len(rows), rating_pass=sum(r['rating_pass'] for r in rows),
        rating_and_votes_pass=sum(r['rating_pass'] and r['votes_pass'] for r in rows),
        previous_metadata_candidates=len(base),
        screen_status_within_previous=dict(Counter(r['screen']['status'] for r in base)),
        previous_unknown_difficulty=sum(r['stratum']['key'] is None for r in base),
        screened_metadata_candidates=len(candidate),
        retained_metadata_unknown_difficulty=sum(r['screen']['status'] == 'eligible_metadata_only'
                                                 and r['stratum']['key'] is None for r in base),
        corpus_phira_ids=len(overlay), corpus_ids_joined=sum(r['current_catalog_audit'] is not None for r in overlay),
        corpus_missing_metadata=sum(r['current_catalog_audit'] is None for r in overlay),
        corpus_train_with_metadata_candidate=sum(r['split'] == 'train' and r['metadata_candidate'] for r in overlay),
        newly_formally_admitted=0)
    inputs = [CATALOG, *MANIFESTS, str(review_path.relative_to(ROOT)),
              'bench/rhythm-v2/fullscale-data-contract-v3.json',
              'beatfcn/ingest/phira_policy.py', 'beatfcn/ingest/coverage.py',
              'beatfcn/ingest/expansion.py', 'beatfcn/train/data.py',
              str(Path(__file__).resolve().relative_to(ROOT))]
    report = dict(schema='beatfcn.phira_strata_audit.v1', updated_at_unix=time.time(),
        strata_version=STRATA_VERSION, screen_version=SCREEN_VERSION, selection_version=PHIRA_SELECTION_VERSION,
        summary=summary, matrices=matrices,
        original_quantity_gate_passed=False, quantity_gate_waived_for_this_run=True,
        formal_training_ready=False, formal_training_started=False, optimizer_updates=0,
        new_cloud_spend_usd=0, wall_seconds=time.monotonic()-started,
        input_sha256={p: digest(ROOT / p) for p in inputs},
        outputs={k: str(p.relative_to(ROOT)) for k, p in paths.items()},
        limitations=['Metadata screening does not verify native playable content, clock or feature tensors.',
                    'Fine strata retain self-reported labels; operational burden must be audited separately.',
                    'Catalog counts are before chart-content deduplication and frozen song splits.',
                    'Full-corpus overlay does not overwrite original manifests or historical benchmark results.'])
    for key, subset in [('.rows.jsonl', rows), ('.quarantine.jsonl', [r for r in rows if not r['metadata_candidate']]),
                        ('.corpus-overlay.jsonl', overlay)]:
        paths[key].write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset))
    report['output_sha256'] = {str(paths[k].relative_to(ROOT)): digest(paths[k])
                              for k in ('.rows.jsonl', '.quarantine.jsonl', '.corpus-overlay.jsonl')}
    paths['.json'].write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    text = ['# Phira 细分层级与清洗首轮结果', '',
        f"冻结目录 {len(rows):,} 条；评分≥{PHIRA_MIN_RATING}且评分人数≥{PHIRA_MIN_RATING_COUNT}、通过原采集条件 {len(base):,} 条。",
        f"这批候选的筛查结果：{summary['screen_status_within_previous']}。",
        f"元数据筛查通过且难度可分层 {len(candidate):,} 条；新正式入训 0。", '',
        '以下是清洗后的元数据候选数，不是有效训练谱数；区间左闭右开，0不入最低档。', '',
        '| 标签 | (0,4) | [4,8) | [8,11) | [11,13) | [13,15) | [15,16) | ≥16 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for tag in ['EZ', 'HD', 'IN', 'AT']:
        counts = matrices['screened_metadata_candidates'].get(tag, {})
        text.append('| ' + tag + ' | ' + ' | '.join(str(counts.get(band, 0)) for _, _, band in LEVEL_BANDS) + ' |')
    text += ['', '每个原始 ID 的分层、排除／待审理由和证据见 `.rows.jsonl`，隔离集合见 `.quarantine.jsonl`。',
             '已收 Phira 谱另有 `.corpus-overlay.jsonl`，绑定旧 manifest 行哈希；原谱和冻结 manifest 均未删除或改写。',
             '14410 因作者自称以 note 表演为主排除；36014／37123 的转换声明进入待审隔离。',
             '14429、11570、56185、72944 的语境复核有固定元数据身份；艺术 drag 不因没有逐颗起音而删除。',
             '本轮数量配额由用户豁免，旧缺额仍真实存在；技术、内容、时钟、特征、拟合与预算检查尚须完成。',
             '新模型更新 0，新增云费用 $0。']
    paths['.md'].write_text('\n'.join(text) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
