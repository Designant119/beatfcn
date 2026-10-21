#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 test 集污染：按歌曲名归一化后重新分组，确保 val 和 test 没有同名歌曲。

策略：
1. 读取 phira_all_charts.jsonl，建立 id -> 归一化歌曲名 的映射
2. 读取 split_full128.json，建立 id -> 集合 的映射
3. 找出所有在 val 和 test 中同时出现的歌曲名（污染）
4. 把 test 中污染的谱面移到 val（保持 test 干净）
5. 生成新的 split_full128_clean.json

歌曲名归一化：去空格、统一大小写、去特殊符号、全角转半角，
尽量把"同一首歌不同上传者的细微命名差异"合并。
"""
import json
from collections import defaultdict
from chart_utils import normalize_name


def main():
    # 1. 读取谱面元数据，建立 id -> name 映射
    id_to_name = {}
    id_to_raw_name = {}
    with open('phira_all_charts.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get('id', ''))
            name = rec.get('name', '')
            if cid:
                id_to_name[cid] = normalize_name(name)
                id_to_raw_name[cid] = name

    print(f'谱面元数据: {len(id_to_name)} 条')

    # 2. 读取当前划分
    with open('split_full128.json', 'r', encoding='utf-8') as f:
        split = json.load(f)

    print(f'当前划分: {len(split)} 个谱面')
    counts = defaultdict(int)
    for v in split.values():
        counts[v] += 1
    print(f'  train={counts["train"]}, val={counts["val"]}, test={counts["test"]}')

    # 3. 按归一化歌名分组，统计每个歌名在各集合中的谱面数
    name_to_ids = defaultdict(list)
    for cid, s in split.items():
        nname = id_to_name.get(cid, '')
        if nname:  # 只统计能找到歌名的
            name_to_ids[nname].append((cid, s))

    print(f'能匹配到歌名的谱面: {sum(len(v) for v in name_to_ids.values())}')
    print(f'不同歌曲名（归一化后）: {len(name_to_ids)}')

    # 4. 找出污染：同一歌名同时出现在 val 和 test
    polluted_names = []
    for nname, ids in name_to_ids.items():
        sets_present = set(s for _, s in ids)
        if 'val' in sets_present and 'test' in sets_present:
            val_ids = [cid for cid, s in ids if s == 'val']
            test_ids = [cid for cid, s in ids if s == 'test']
            polluted_names.append((nname, val_ids, test_ids))

    print(f'\n=== 污染分析 ===')
    print(f'同时出现在 val 和 test 的歌曲名: {len(polluted_names)} 个')
    total_test_polluted = sum(len(t) for _, _, t in polluted_names)
    print(f'test 中被污染的谱面数: {total_test_polluted}')
    test_total = counts['test']
    print(f'test 污染比例: {total_test_polluted}/{test_total} = {total_test_polluted/test_total*100:.1f}%')

    # 列出前10个污染歌名
    print(f'\n污染最严重的前10首歌:')
    polluted_names.sort(key=lambda x: -(len(x[1]) + len(x[2])))
    for nname, val_ids, test_ids in polluted_names[:10]:
        raw = id_to_raw_name.get(val_ids[0], nname)
        print(f'  {raw}: val={len(val_ids)}个, test={len(test_ids)}个')

    # 5. 修复：把 test 中污染的谱面移到 val
    new_split = dict(split)  # 复制
    moved = 0
    for nname, val_ids, test_ids in polluted_names:
        for cid in test_ids:
            new_split[cid] = 'val'
            moved += 1

    print(f'\n=== 修复结果 ===')
    print(f'从 test 移到 val 的谱面数: {moved}')

    new_counts = defaultdict(int)
    for v in new_split.values():
        new_counts[v] += 1
    print(f'修复后: train={new_counts["train"]}, val={new_counts["val"]}, test={new_counts["test"]}')

    # 6. 验证修复后没有污染
    name_to_ids_new = defaultdict(list)
    for cid, s in new_split.items():
        nname = id_to_name.get(cid, '')
        if nname:
            name_to_ids_new[nname].append(s)

    remaining_pollution = 0
    for nname, sets in name_to_ids_new.items():
        if 'val' in sets and 'test' in sets:
            remaining_pollution += 1

    print(f'修复后 val/test 同名歌曲数: {remaining_pollution}（应为0）')

    # 7. 保存新划分
    out_file = 'split_full128_clean.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(new_split, f, ensure_ascii=False, indent=2)
    print(f'\n已保存干净划分到: {out_file}')
    print(f'使用方法: 训练时加 --split {out_file}')

    # 8. 同时生成污染报告
    report_file = 'test_pollution_report.json'
    report = {
        'original_test_count': test_total,
        'polluted_test_count': total_test_polluted,
        'pollution_ratio': total_test_polluted / test_total,
        'polluted_song_count': len(polluted_names),
        'moved_to_val': moved,
        'clean_test_count': new_counts['test'],
        'polluted_songs': [
            {
                'normalized_name': nname,
                'raw_name': id_to_raw_name.get(val_ids[0], ''),
                'val_ids': val_ids,
                'test_ids': test_ids,
            }
            for nname, val_ids, test_ids in polluted_names
        ],
    }
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'污染详情已保存到: {report_file}')


if __name__ == '__main__':
    main()
