"""Snapshot council sources and render the decision history. No model/cloud calls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REL = Path('reports/council-archive')
STATUSES = {
    'adopted': '已采用', 'partial': '改动后采用／部分采用',
    'deferred': '暂缓', 'rejected': '不采用',
    'tested_supported': '已测试，有限定支持',
    'tested_failed': '已测试，未通过', 'gap': '实现偏离／待处理',
}


def read_json(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def local(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Outside repository: {name}')
    return path


def validate_events(rows):
    seen = set()
    for row in rows:
        for key in ('event_id', 'id', 'title', 'attempt_id', 'source', 'source_sha256',
                    'locator', 'status', 'recommendation', 'rationale', 'evidence',
                    'revisit', 'recorded_at', 'recording_basis', 'context'):
            if key not in row or row[key] in (None, '', []):
                raise ValueError(f'Missing decision field {key}: {row.get("id")}')
        if row['status'] not in STATUSES or row['event_id'] in seen:
            raise ValueError(f'Invalid status or duplicate event: {row["event_id"]}')
        seen.add(row['event_id'])
    return rows


def events(root):
    path = root / REL / 'decisions.jsonl'
    return validate_events([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def append_event(root, row):
    old = events(root)
    validate_events(old + [row])
    manifest = read_json(root / REL / 'manifest.json')
    attempts = {a['id']: a for a in manifest['attempts']}
    if row['attempt_id'] not in attempts:
        raise ValueError('Unknown source attempt')
    if digest(local(root, row['source']).read_bytes()) != row['source_sha256']:
        catalog = read_json(root / REL / 'catalog.json')
        if not any(v['source'] == row['source'] and v['sha256'] == row['source_sha256']
                   for v in catalog['versions']):
            raise ValueError('Unknown historical opinion version')
    for path in row['evidence']:
        if not local(root, path).is_file():
            raise ValueError(f'Missing evidence: {path}')
    # All validation precedes the append; a rejected event cannot poison history.
    with (root / REL / 'decisions.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def snapshot(root):
    root = root.resolve()
    base = root / REL
    manifest = read_json(base / 'manifest.json')
    catalog_path = base / 'catalog.json'
    catalog = read_json(catalog_path) if catalog_path.exists() else {'schema_version': 1, 'versions': []}
    # Validate existing objects before adding anything; never repair a corrupt snapshot in place.
    for version in catalog['versions']:
        if digest(local(root, version['object']).read_bytes()) != version['sha256']:
            raise ValueError(f'Corrupt historical object: {version["object"]}')
    prior_logs = [v for v in catalog['versions'] if v['source'] == str(REL / 'decisions.jsonl')]
    if prior_logs:
        previous = local(root, prior_logs[-1]['object']).read_bytes()
        if not (base / 'decisions.jsonl').read_bytes().startswith(previous):
            raise ValueError('Decision history was rewritten; append a new event')
    known = {(v['source'], v['sha256']) for v in catalog['versions']}
    sources = {}
    for round_ in manifest['rounds']:
        for path in sorted(local(root, round_['source_dir']).iterdir()):
            if path.is_file():
                sources[str(path.relative_to(root))] = round_['id']
    for attempt in manifest['attempts']:
        for key in ('receipt_files', 'prompt_files'):
            for path in attempt.get(key, []):
                sources[path] = attempt['round_id']
    for name in manifest.get('outcome_evidence', []):
        sources[name] = 'outcomes'
    for name in manifest.get('workflow_files', []):
        sources[name] = 'workflow'
    now = datetime.now(timezone.utc).isoformat()
    for name, round_id in sources.items():
        path = local(root, name)
        data = path.read_bytes()
        sha = digest(data)
        if (name, sha) in known:
            continue
        target = REL / 'objects' / f'{sha}{path.suffix}'
        dest = local(root, target)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.read_bytes() != data:
                raise ValueError(f'Object collision: {target}')
        else:
            with dest.open('xb') as stream:
                stream.write(data)
        catalog['versions'].append(dict(source=name, sha256=sha, object=str(target),
            bytes=len(data), round_id=round_id, captured_at=now,
            recording_basis=manifest['recording_basis']))
    # Preserve the manifest and decision log at each revision as well.
    for name in ('manifest.json', 'decisions.jsonl'):
        path = base / name
        data = path.read_bytes()
        sha = digest(data)
        source = str(path.relative_to(root))
        if (source, sha) not in known:
            target = REL / 'objects' / f'{sha}{path.suffix}'
            dest = root / target
            if not dest.exists():
                with dest.open('xb') as stream:
                    stream.write(data)
            elif dest.read_bytes() != data:
                raise ValueError(f'Object collision: {target}')
            catalog['versions'].append(dict(source=source, sha256=sha, object=str(target),
                bytes=len(data), round_id='registry', captured_at=now,
                recording_basis=manifest['recording_basis']))
    write_json(catalog_path, catalog)
    return manifest, catalog


def check(root):
    base = root / REL
    manifest = read_json(base / 'manifest.json')
    catalog = read_json(base / 'catalog.json')
    attempts = {a['id']: a for a in manifest['attempts']}
    if len(attempts) != len(manifest['attempts']):
        raise ValueError('Duplicate attempt IDs')
    versions = {}
    for item in catalog['versions']:
        if digest(local(root, item['object']).read_bytes()) != item['sha256']:
            raise ValueError(f'Changed archive object: {item["object"]}')
        versions[(item['source'], item['sha256'])] = item
    for attempt in attempts.values():
        if attempt['outcome'] not in ('pending', 'completed', 'failed', 'interrupted'):
            raise ValueError(f'Unknown outcome: {attempt["id"]}')
        if attempt['outcome'] == 'pending':
            if attempt.get('report') or attempt.get('receipt_files'):
                raise ValueError('Pending attempt must not claim a result')
            continue
        if not attempt.get('receipt_files'):
            raise ValueError(f'No terminal receipt: {attempt["id"]}')
        receipt = read_json(local(root, attempt['receipt_files'][0]))['result']['structuredContent']
        if receipt['taskId'] != attempt['task_id'] or receipt['hasPendingChildRuns']:
            raise ValueError(f'Receipt identity/liveness: {attempt["id"]}')
        if receipt['model'] != attempt['model'] or receipt.get('options', []) != attempt.get('options', []):
            raise ValueError(f'Model/settings mismatch: {attempt["id"]}')
        if attempt['outcome'] == 'completed':
            if receipt['status'] != 'completed' or not attempt.get('report'):
                raise ValueError(f'Missing completed opinion: {attempt["id"]}')
        if attempt.get('report') and not any(s == attempt['report'] for s, _ in versions):
            raise ValueError(f'Unsnapshotted report: {attempt["report"]}')
        if not attempt.get('prompt_complete'):
            raise ValueError(f'Incomplete prompt archive: {attempt["id"]}')
        for path in [attempt.get('report')] + attempt['receipt_files'] + attempt['prompt_files']:
            if path and (path, digest(local(root, path).read_bytes())) not in versions:
                raise ValueError(f'Unarchived current source: {path}')
        prompt, raw, *continuations = attempt['prompt_files']
        first = read_json(local(root, raw))['result']['structuredContent']['items'][0]
        chunks = [first['text']]
        last = first
        for path in continuations:
            for page in read_json(local(root, path))['pages']:
                last = next(i for i in page['structuredContent']['items'] if i['itemId'] == first['itemId'])
                chunks.append(last['text'])
        if last['textTruncated'] or ''.join(chunks) != local(root, prompt).read_text():
            raise ValueError(f'Prompt pagination incomplete: {attempt["id"]}')
    rows = events(root)
    for row in rows:
        if row['attempt_id'] not in attempts:
            raise ValueError(f'Unknown attempt: {row["attempt_id"]}')
        if (row['source'], row['source_sha256']) not in versions:
            raise ValueError(f'Missing exact opinion version: {row["id"]}')
        for name in row['evidence']:
            if not local(root, name).is_file():
                raise ValueError(f'Missing evidence: {name}')
    current = {r['id']: r for r in rows}
    return dict(attempts=dict(Counter(a['outcome'] for a in attempts.values())),
        recommendation_count=len(current), decision_events=len(rows),
        source_versions=len(versions), statuses=dict(Counter(r['status'] for r in current.values())))


def verify_backfill(root):
    baseline = read_json(root / REL / 'backfill-baseline.json')
    for path, sha in baseline['sha256'].items():
        if path not in baseline['allowed_to_change'] and digest(local(root, path).read_bytes()) != sha:
            raise ValueError(f'Original source changed during backfill: {path}')
    before = root / REL / 'inputs/AGENTS-before-archive.md'
    if not (root / 'AGENTS.md').read_bytes().startswith(before.read_bytes()):
        raise ValueError('Existing AGENTS instructions were altered')
    return len(baseline['sha256'])


def link(root, name, label=None):
    # Views live at reports/; paths are relative to that directory.
    path = Path(name)
    target = str(path.relative_to('reports')) if path.parts[0] == 'reports' else '../' + str(path)
    return f'[{label or path.name}]({target})'


def render(root, manifest, catalog):
    rows = events(root)
    current = {r['id']: r for r in rows}
    versions = {(v['source'], v['sha256']): v for v in catalog['versions']}
    stats = check(root)
    lines = ['# Council 档案索引', '',
        '初始历史为事后补录，后续完成回执到达即追加。原文与回执按字节保存；'
        '取舍按实际记录时间追溯，不能冒充先前已有的决策。', '',
        f'已收到 {stats["attempts"].get("completed", 0)} 份完整意见；待回收 '
        f'{stats["attempts"].get("pending", 0)} 个任务；失败／取消 '
        f'{stats["attempts"].get("failed", 0) + stats["attempts"].get("interrupted", 0)} 次。'
        f'索引 {len(current)} 条行动建议、{len(rows)} 条状态记录。', '',
        '[暂缓与待处理建议](COUNCIL_BACKLOG.md) · [全部建议与历史](COUNCIL_DECISIONS.md) · '
        '[当前策略差异](COUNCIL_STRATEGY_AUDIT.md) · [归档规则](../design/COUNCIL_ARCHIVE.md)', '',
        '原报告、输入包、父级裁决分别保存。完整原文是意见权威；建议索引是父级整理，不能代替原文。'
        '所有独立报告中的限制和反对意见也保留，不以多数票当作独立实验证据。', '']
    for round_ in manifest['rounds']:
        lines += [f'## {round_["id"]} · {round_["title"]}', '', round_['context'], '',
            '| 评审／尝试 | 模型与档位 | 状态 | 原文／回执／输入 |', '|---|---|---|---|']
        for a in manifest['attempts']:
            if a['round_id'] != round_['id']:
                continue
            files = []
            if a.get('report'):
                files.append(link(root, a['report'], '原始位置'))
            for path in [a.get('report')] + a.get('receipt_files', []) + a.get('prompt_files', []):
                if path:
                    vv = [v for v in catalog['versions'] if v['source'] == path][-1]
                    files.append(link(root, vv['object'], Path(path).name))
            lines.append(f'| {a["id"]} {a["label"]} | {a["model"]} / {a["effort"]} | '
                f'{a["outcome"]}: {a["note"]} | {" · ".join(files) or "尚无结果，不轮询"} |')
        lines += ['', '当时材料和父级记录（本次快照）：', '']
        seen = {}
        for v in catalog['versions']:
            if v['round_id'] == round_['id'] and v['source'].startswith(round_['source_dir'] + '/'):
                seen[v['source']] = v
        for name, v in seen.items():
            lines.append(f'- {link(root, name, Path(name).name)} · '
                f'{link(root, v["object"], "冻结副本")} · SHA256 `{v["sha256"]}`')
        lines += ['']
    lines += ['## 完整性与缺口', '',
        f'{sum(a["outcome"] != "pending" for a in manifest["attempts"])} 次已终止任务均回收最终回执和完整首条任务提示；'
        '长提示已按工具偏移分页拼接。未返回任务只登记派发清单及已有现场材料；未编造最终意见。'
        '原始 502 的工具状态为 completed，'
        '但其内容是执行错误，因此记作 failed，不计入有效报告。', '',
        '访问限制记录反映可见工具轨迹，不是操作系统隔离证明。原文快照内的相对链接以原始位置为基准；'
        '阅读相关链接可使用各报告的“原始位置”。报告内的外部论文保留原链接；'
        '本次没有重新抓取全部论文或历史云 trace。', '',
        '机器入口：' + link(root, str(REL / 'manifest.json')) + ' · '
        + link(root, str(REL / 'catalog.json')) + ' · ' + link(root, str(REL / 'decisions.jsonl')), '']
    (root / 'reports/COUNCIL_INDEX.md').write_text('\n'.join(lines))
    for backlog, name in ((False, 'COUNCIL_DECISIONS.md'), (True, 'COUNCIL_BACKLOG.md')):
        selected = [r for r in current.values() if not backlog or r['status'] in ('deferred', 'partial', 'gap')]
        out = ['# ' + ('暂缓与待处理建议池' if backlog else 'Council 建议与取舍历史'), '',
            '此视图由 decisions.jsonl 生成。恢复建议时追加同一 ID 的决策记录，保留先前理由与实验身份。', '',
            '[全部轮次](COUNCIL_INDEX.md) · [当前策略差异](COUNCIL_STRATEGY_AUDIT.md)', '']
        for r in selected:
            v = versions[(r['source'], r['source_sha256'])]
            out += [f'## {r["id"]} · {r["title"]}', '',
                f'状态：**{STATUSES[r["status"]]}**；来源：{r["attempt_id"]}，'
                f'{link(root, v["object"], r["locator"])}。', '',
                f'原建议：{r["recommendation"]}', '', f'本次取舍／事实：{r["rationale"]}', '',
                '实现／结果证据：' + ' · '.join(link(root, p) for p in r['evidence']), '',
                f'回收条件／下一检查：{r["revisit"]}', '',
                f'记录性质：{r["recording_basis"]}；对应现场：{r["context"]}。', '']
            history = [e for e in rows if e['id'] == r['id']]
            if len(history) > 1:
                out += ['状态历史（区分回溯补录与后续新决策）：', '']
                for e in history:
                    old = versions[(e['source'], e['source_sha256'])]
                    out.append(f'- {e["event_id"]} / {e["recorded_at"]} / {e["context"]} / '
                        f'{STATUSES[e["status"]]} / {link(root, old["object"], e["attempt_id"])}：{e["rationale"]}')
                out += ['']
        (root / 'reports' / name).write_text('\n'.join(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sync', 'check', 'record'))
    parser.add_argument('--verify-backfill', action='store_true',
                        help='One-time check against the 2026-10-27 pre-edit baseline')
    args = parser.parse_args()
    if args.action == 'record':
        append_event(ROOT, json.load(sys.stdin))
    if args.action in ('sync', 'record'):
        manifest, catalog = snapshot(ROOT)
        render(ROOT, manifest, catalog)
    result = check(ROOT)
    if args.verify_backfill:
        result['backfill_original_files_checked'] = verify_backfill(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
