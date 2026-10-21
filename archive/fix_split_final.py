#!/usr/bin/env python3
"""彻底清洗split：把train里和val/test同名的歌全部挪到val，确保零泄漏。
使用 chart_utils.normalize_name（含NFKC），和 fix_test_split.py 保持一致。"""
import json, sys
from chart_utils import normalize_name

# 加载元数据
meta = {}
meta_raw = {}  # 原始歌名，归一化后为空时用
with open('phira_all_charts.jsonl') as f:
    for line in f:
        d = json.loads(line)
        sid = str(d['id'])
        meta[sid] = normalize_name(d.get('name', ''))
        meta_raw[sid] = d.get('name', '')

# 加载当前split（从final版继续修，而不是clean版）
with open('split_full128_final.json') as f:
    split = json.load(f)

# 按集合分组
train_ids = [i for i, s in split.items() if s == 'train']
val_ids = [i for i, s in split.items() if s == 'val']
test_ids = [i for i, s in split.items() if s == 'test']

def name_of(i):
    return meta.get(i, '')

# 收集val和test里的所有歌名（归一化后的，和原始的）
val_names = set()
test_names = set()
raw_val_names = {}  # 归一化名空的，用原始名比较
raw_test_names = {}
for i in val_ids:
    n = name_of(i)
    if n:
        val_names.add(n)
    else:
        # 归一化后为空的（比如全是符号的歌名），用原始名
        raw = meta_raw.get(i, '')
        if raw:
            raw_val_names[raw] = i
for i in test_ids:
    n = name_of(i)
    if n:
        test_names.add(n)
    else:
        raw = meta_raw.get(i, '')
        if raw:
            raw_test_names[raw] = i

# 找出train里和val/test同名的歌
leak_to_val = []
leak_to_test = []
for i in train_ids:
    n = name_of(i)
    if n:
        if n in test_names:
            leak_to_test.append(i)
        elif n in val_names:
            leak_to_val.append(i)
    else:
        # 归一化后为空的，用原始名比较
        raw = meta_raw.get(i, '')
        if raw:
            if raw in raw_test_names:
                leak_to_test.append(i)
            elif raw in raw_val_names:
                leak_to_val.append(i)

print(f'train总数: {len(train_ids)}')
print(f'  和val同名: {len(leak_to_val)} 首 → 挪到val')
print(f'  和test同名: {len(leak_to_test)} 首 → 挪到test')

# 挪动
for i in leak_to_val:
    split[i] = 'val'
for i in leak_to_test:
    split[i] = 'test'

# 重新统计
train_ids2 = [i for i, s in split.items() if s == 'train']
val_ids2 = [i for i, s in split.items() if s == 'val']
test_ids2 = [i for i, s in split.items() if s == 'test']

train_n = set(name_of(i) for i in train_ids2 if name_of(i))
val_n = set(name_of(i) for i in val_ids2 if name_of(i))
test_n = set(name_of(i) for i in test_ids2 if name_of(i))

print(f'\n清洗后:')
print(f'train: {len(train_ids2)} 首, {len(train_n)} 个歌名')
print(f'val: {len(val_ids2)} 首, {len(val_n)} 个歌名')
print(f'test: {len(test_ids2)} 首, {len(test_n)} 个歌名')
print(f'\n泄漏检查（归一化歌名）:')
print(f'  train/val 同曲: {len(train_n & val_n)} 个')
print(f'  train/test 同曲: {len(train_n & test_n)} 个')
print(f'  val/test 同曲: {len(val_n & test_n)} 个')

# 额外检查：空歌名的（归一化后为空，比如全是符号的歌名）
def raw_names(ids):
    return {meta_raw.get(i,'') for i in ids if meta_raw.get(i,'') and not name_of(i)}

train_raw = raw_names(train_ids2)
val_raw = raw_names(val_ids2)
test_raw = raw_names(test_ids2)
print(f'\n泄漏检查（空归一化名，用原始名）:')
print(f'  train/val 同曲: {len(train_raw & val_raw)} 个')
print(f'  train/test 同曲: {len(train_raw & test_raw)} 个')
print(f'  val/test 同曲: {len(val_raw & test_raw)} 个')
if val_raw & test_raw:
    print(f'  ⚠️ val/test空歌名同曲: {val_raw & test_raw}')

# 保存
with open('split_full128_final.json', 'w', encoding='utf-8') as f:
    json.dump(split, f, ensure_ascii=False, indent=1)
print(f'\n已保存 split_full128_final.json')
