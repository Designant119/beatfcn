#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeatFCN v22 核心函数单元测试。
运行：python test_core.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from chart_utils import normalize_name
from model import compute_flux, BeatFCNContext


def test_normalize_name():
    """测试歌名归一化：NFKC全角转半角、去空格、去标点、小写。"""
    # 全角括号 vs 半角括号 → 应该归一化后相同
    assert normalize_name('Song（Remix）') == normalize_name('Song(Remix)')
    # 大小写
    assert normalize_name('ABC') == normalize_name('abc')
    # 空格
    assert normalize_name('A B C') == normalize_name('ABC')
    # 中日韩文字保留
    assert '望影' in normalize_name('望影の方舟')
    # 空字符串
    assert normalize_name('') == ''
    assert normalize_name(None) == ''
    print('✅ normalize_name 测试通过')


def test_compute_flux():
    """测试谱通量计算：形状正确、值在合理范围。"""
    mel = np.random.randn(100, 81).astype(np.float32)
    flux = compute_flux(mel)
    assert flux.shape == (100, 1), f'形状错误: {flux.shape}'
    assert flux.dtype == np.float32
    assert flux.min() >= 0, 'flux应该非负'
    assert flux.max() <= 2, 'flux应该clip到2'
    print('✅ compute_flux 测试通过')


def test_model_forward():
    """测试模型前向传播：输出形状正确。"""
    model = BeatFCNContext(n_mels=81, ch=32, n_pitch=46, n_layers=2, nhead=2)
    model.eval()
    B, T, G = 2, 500, 100
    mel = torch.randn(B, T, 81 + 46)
    grid = torch.linspace(0, 10, G).unsqueeze(0).expand(B, -1)
    notes = torch.full((B, G), -1, dtype=torch.float)  # 全遮罩
    with torch.no_grad():
        o1, o2 = model(mel, grid, notes)
    assert o1.shape == (B, G), f'o1形状错误: {o1.shape}'
    assert o2.shape == (B, G), f'o2形状错误: {o2.shape}'
    print('✅ BeatFCNContext 前向传播测试通过')


def test_model_params():
    """测试模型参数量：和文档一致（658,946）。"""
    model = BeatFCNContext()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 658946, f'参数量错误: {n_params}, 期望658946'
    print(f'✅ 模型参数量正确: {n_params:,}')


def test_no_leakage():
    """测试split_final无泄漏：train/val/test同曲=0。"""
    import json
    with open('split_full128_final.json') as f:
        split = json.load(f)
    with open('phira_all_charts.jsonl') as f:
        meta = {str(json.loads(l)['id']): normalize_name(json.loads(l).get('name','')) for l in f}

    def names(s):
        return set(meta.get(i,'') for i, v in split.items() if v == s and meta.get(i,''))

    train_n, val_n, test_n = names('train'), names('val'), names('test')
    assert len(train_n & val_n) == 0, f'train/val同曲: {len(train_n & val_n)}'
    assert len(train_n & test_n) == 0, f'train/test同曲: {len(train_n & test_n)}'
    assert len(val_n & test_n) == 0, f'val/test同曲: {len(val_n & test_n)}'
    print(f'✅ split无泄漏: train={len([i for i,v in split.items() if v=="train"])}, '
          f'val={len([i for i,v in split.items() if v=="val"])}, '
          f'test={len([i for i,v in split.items() if v=="test"])}')


if __name__ == '__main__':
    import torch
    test_normalize_name()
    test_compute_flux()
    test_model_forward()
    test_model_params()
    test_no_leakage()
    print('\n🎉 全部测试通过！')
