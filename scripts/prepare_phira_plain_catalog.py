"""D29 metadata-only supply queue; no chart is declared human-verified or admitted."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from beatfcn.events.group_split import work_key
from beatfcn.ingest.coverage import native_tier, phira_native_difficulty
from beatfcn.ingest.expansion import tag_of, STANDARD_TAGS

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'c0ed193538858d69d2e1c02e79465bf1054c6fd2'


def main():
    index = ROOT/'inputs/data_scale_v2/phira_plain_discovery/archive-index-c0ed193.jsonl'
    receipt = json.loads((ROOT/'reports/d29_plain_index_receipt.json').read_text())
    if hashlib.sha256(index.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('archive identity changed')
    old_path = ROOT/'inputs/data_scale_v2/catalog_v1/phira/catalog.jsonl'
    if hashlib.sha256(old_path.read_bytes()).hexdigest() != '448b623e931b200388bfb78f4b5050b797c851c5fbffa537663d9e69956cf7e2':
        raise ValueError('regular snapshot changed')
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    old_ids = {int(r['id']) for r in old}
    policy = json.loads((ROOT/'bench/rhythm-v2/data-scale-policy-v1.json').read_text())
    old_works = {work_key(r.get('name')) for r in old if r.get('name')}
    rows = []
    for line in index.read_text().splitlines():
        row = json.loads(line)
        if row['id'] in old_ids or 'plain' not in row.get('divisions', []):
            continue
        number, reason = phira_native_difficulty(row)
        tier = native_tier('phira', number, policy)
        rows.append(dict(row, native_difficulty=number, native_difficulty_evidence=reason,
            difficulty_tier=tier, standard_tag=tag_of(row) in STANDARD_TAGS,
            title_in_regular=work_key(row.get('name')) in old_works,
            metadata_url=f'https://raw.githubusercontent.com/swordalt/phira-archive/{COMMIT}/data/charts/{row["id"]//1000}/{row["id"]}.json',
            provenance_status='full_metadata_and_native_human_authorship_pending',
            training_ready=False))
    queue, works = [], set()
    for tier in ('low', 'medium'):
        candidates = [r for r in rows if r['difficulty_tier']==tier and r['standard_tag'] and work_key(r.get('name'))]
        candidates.sort(key=lambda r:(r['title_in_regular'], hashlib.sha256(f'20261025:{r["id"]}'.encode()).hexdigest()))
        selected = 0
        for row in candidates:
            work = work_key(row['name'])
            if work in works:
                continue
            queue.append(row)
            works.add(work)
            selected += 1
            if selected == 20:
                break
    output = index.parent/'plain-candidates.jsonl'
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report = dict(schema='beatfcn.phira_plain_metadata_supply.v1', complete=True,
        source_sha256=hashlib.sha256(index.read_bytes()).hexdigest(), candidate_file=str(output.relative_to(ROOT)),
        candidate_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), new_plain_ids=len(rows),
        tiers=dict(Counter(r['difficulty_tier'] or 'unknown' for r in rows)),
        standard_low_medium=dict(Counter(r['difficulty_tier'] for r in rows if r['standard_tag'] and r['difficulty_tier'] in ('low','medium'))),
        metadata_review_queue=queue, effective_training_charts_added=0, chart_archives_downloaded=0,
        new_paid_compute_usd=0, training_ready=False,
        limitations=['An archive ID is not an independent target; different uploads, AutoSave drafts, converted or generated charts cannot fill quotas.',
            'No full metadata has yet been fetched by this script; manual-native provenance and URLs remain pending.',
            '20 per tier is a metadata-review queue, not a selected training set. Nonstandard tags are retained in the catalog for separate review.',
            'Source metadata is a pinned archive snapshot; current live availability has not been re-queried.'])
    (ROOT/'reports/d29_plain_catalog.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('metadata_review_queue','limitations')},ensure_ascii=False))


if __name__ == '__main__':
    main()
