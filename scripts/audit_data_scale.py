"""Inventory source supply and candidate gaps without declaring training readiness."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.ingest.coverage import native_tier, phira_native_difficulty, policy_identity, quota_report
from beatfcn.ingest.expansion import phira_acquisition_eligible
from beatfcn.train.data import (selected_for_training, dup_only, PHIRA_MIN_RATING,
                               PHIRA_MIN_RATING_COUNT, PHIRA_SELECTION_VERSION, phira_rating_metadata)
from beatfcn.ingest.phira_policy import screen_metadata


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default='reports/data_scale_audit')
    args = parser.parse_args()
    output = ROOT / args.output_prefix
    if output.with_suffix('.json').exists() or output.with_suffix('.md').exists():
        raise FileExistsError('Use a new output prefix to preserve prior audits')
    policy_path = ROOT / 'bench/rhythm-v2/data-scale-policy-v1.json'
    policy = json.loads(policy_path.read_text())
    paths = [policy_path,
             ROOT / 'inputs/data_scale_v2/catalog_mania_v2/mania/catalog.jsonl',
             ROOT / 'inputs/data_scale_v2/catalog_v1/phira/catalog.jsonl',
             ROOT / 'inputs/rhythm_v2_preparation/charts.offset-v2-3d99ff6bd5e7.jsonl',
             ROOT / 'inputs/data_expansion_v1/corpus/manifest/expansion-candidates.jsonl',
             ROOT / 'inputs/data_scale_v2/discovery/phigros_difficulty.tsv']
    mania_sets, phira = lines(paths[1]), lines(paths[2])
    mania = {str(b['id']): b for r in mania_sets for b in r['beatmaps']
             if b.get('mode_int') == 3 and b.get('cs') == 4 and b.get('status') in ('ranked', 'approved')}
    phira_by_id = {str(r['id']): r for r in phira}
    metadata = {'mania_4k': [], 'phira': []}
    for row in mania.values():
        metadata['mania_4k'].append(native_tier('mania_4k', row.get('difficulty_rating'), policy) or 'unknown')
    for row in phira:
        if phira_acquisition_eligible(row):
            value, reason = phira_native_difficulty(row)
            metadata['phira'].append(native_tier('phira', value, policy) or 'unknown')
    supply = {m: dict(Counter(values)) for m, values in metadata.items()}
    rating_scenarios = {}
    for name, minimum, votes in [('previous_090_100votes', .90, 100),
                                  ('085_100votes_comparison', .85, 100),
                                  ('previous_085_no_vote_floor', .85, None),
                                  ('current_075_50votes', PHIRA_MIN_RATING, PHIRA_MIN_RATING_COUNT)]:
        counts = Counter()
        for row in phira:
            metadata_row = dict(row, source='phira', difficulty_label=row.get('level'))
            if (not phira_acquisition_eligible(row)
                    or screen_metadata(metadata_row)['status'] != 'eligible_metadata_only'):
                continue
            rating, count = phira_rating_metadata(row)
            if rating is None or rating < minimum or (votes is not None and (count is None or count < votes)):
                continue
            value, _ = phira_native_difficulty(row)
            counts[native_tier('phira', value, policy) or 'unknown'] += 1
        rating_scenarios[name] = dict(counts)
    official = Counter()
    for line in paths[5].read_text().splitlines():
        for value in line.split('\t')[1:]:
            try:
                tier = native_tier('phira', float(value), policy)
            except ValueError:
                tier = None
            official[tier or 'unknown'] += 1
    old, new = lines(paths[3]), lines(paths[4])
    annotated = []
    for row in old + new:
        r = dict(row)
        mode = r.get('mode')
        if mode not in ('mania_4k', 'phira'):
            continue
        if mode == 'mania_4k':
            value = mania.get(r['native_id'], {}).get('difficulty_rating')
            evidence = 'current_native_id_join; historical chart checksum verification pending'
        else:
            value, evidence = phira_native_difficulty(phira_by_id.get(r['native_id'], r))
        r.update(difficulty_tier=native_tier(mode, value, policy), native_difficulty=value,
                 difficulty_evidence=evidence, difficulty_review_status='unverified',
                 difficulty_policy_sha256=policy_identity(policy))
        annotated.append(r)
    old_ids = {r['chart_id'] for r in old}
    potential = {}
    for mode in ('mania_4k', 'phira'):
        for tier in policy['tiers'] + ['unknown']:
            rows = [r for r in annotated if r.get('mode') == mode and (r['difficulty_tier'] or 'unknown') == tier
                    and r.get('split') == 'train']
            prior = [r for r in rows if r['chart_id'] in old_ids and selected_for_training(r)
                     and (r.get('kept') or dup_only(r))]
            additions = [r for r in rows if r['chart_id'] not in old_ids and r.get('technical_audit_pass')
                         and selected_for_training(r)]
            potential[f'{mode}/{tier}'] = dict(existing_selected_train=len(prior),
                d18_technical_train=len(additions), candidate_sum_before_content_dedup=len(prior) + len(additions),
                unique_audio=len({r['audio_sha256'] for r in prior + additions}),
                known_minimum_shortfall=max(0, policy['minimum_unique_train_charts_per_tier'][mode] - len(prior) - len(additions)) if tier != 'unknown' else None)
    gate = quota_report(annotated, policy)
    report = dict(schema='beatfcn.data_scale_audit.v1', updated_at_unix=time.time(),
        policy=policy, policy_sha256=policy_identity(policy),
        selection_policy=dict(id=PHIRA_SELECTION_VERSION, minimum_rating=PHIRA_MIN_RATING,
                              minimum_votes=PHIRA_MIN_RATING_COUNT, votes_role='eligibility_and_confidence'),
        selection_code_sha256=hashlib.sha256((ROOT / 'beatfcn/train/data.py').read_bytes()).hexdigest(),
        inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        public_catalog_supply=supply, public_mania_sets=len(mania_sets),
        phira_catalog_rating_scenarios=rating_scenarios,
        additional_phigros_archive_metadata=dict(official),
        official_metadata_source='https://github.com/7aGiven/Phigros_Resource/tree/da34d1de07cd59b096ca0520b2815a2ea11cc13e',
        candidate_train_upper_bounds=potential, formal_admission_audit=gate,
        formal_training_ready=False, formal_training_started=False,
        limitations=['Source counts are before target dedup, training split, technical validation and feature encoding.',
                     'Official archive counts can overlap Phira uploads and cannot simply be added to accepted counts.',
                     'Historical mania labels joined by ID need file-checksum verification; native Phira labels need burden review.',
                     'Zero verified admission means the new contract has not been certified, not that previous training files disappeared.',
                     'Manifest candidate counts use their stored ratings; current catalog scores are not silently joined into frozen manifests.',
                     'Source availability may fall short of the requested Phira low-tier quota; do not fill with duplicate/augmented charts.'])
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    text = ['# 三层数据规模盘点', '', '每层最低 mania 1,500、Phira 700；正式训练仍未就绪。', '',
            '以下为训练候选的上界，尚未完成新合同的内容去重、难度复核和特征验收。', '',
            '| 模式 / 层 | 现有训练候选 | D18 新增技术候选 | 至少还缺 |', '|---|---:|---:|---:|']
    for key, value in potential.items():
        if key.endswith('/unknown'):
            continue
        text.append(f'| {key} | {value["existing_selected_train"]} | {value["d18_technical_train"]} | {value["known_minimum_shortfall"]} |')
    text += ['', '公开目录的原生难度分布（未解析，不是入训数）：', '',
             '| 来源 | 低 | 中 | 高 | 未知/冲突 |', '|---|---:|---:|---:|---:|']
    for name, counts in [*supply.items(), ('Phigros 资源档案（额外候选）', official)]:
        text.append('| ' + name + ' | ' + ' | '.join(str(counts.get(t, 0)) for t in ['low','medium','high','unknown']) + ' |')
    text += ['', 'Phira 目录评分门对照（包含既有谱、尚未扣划分/去重/技术失败）：', '',
             '| 评分规则 | 低 | 中 | 高 | 未知/冲突 |', '|---|---:|---:|---:|---:|']
    for name, counts in rating_scenarios.items():
        text.append('| ' + name + ' | ' + ' | '.join(str(counts.get(t, 0)) for t in ['low','medium','high','unknown']) + ' |')
    text += ['', '训练候选使用各自 manifest 中保存的评分；不把最新目录评分静默回填到冻结清单。',
             'Phigros 档案与 Phira 可能重复，来源数量不能直接相加；低难度源仍存在实质缺口。',
             '公开元数据采集不消耗新云计算额度；模型质量、正式训练曝光和预算要求保持。', '']
    output.with_suffix('.md').write_text('\n'.join(text))
    print(json.dumps(dict(supply=supply, official=dict(official), candidates=potential,
                         rating_scenarios=rating_scenarios), ensure_ascii=False))


if __name__ == '__main__':
    main()
