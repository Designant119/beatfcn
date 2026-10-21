#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BeatFCN v22 Top-K 全量重评。
用法: python3 reval_v22.py --data ./grid_npz_full128 --split ./split_full128_final.json --pitch-dir ./pitch_npz_full
复用 beat_fcn_train_v22 的 Songs / BeatFCNContext / f1_inst / 常量。
"""
import os, sys, json, glob, time, argparse, random
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beat_fcn_train_v22 as T  # noqa: E402

TH1, TH2 = 0.4, 0.5
TOLS_MS = [0, 11, 22, 50]
TOLS_S = [m / 1000.0 for m in TOLS_MS]


def load_model(ckpt_path, ch=96, n_pitch=46, device='cpu'):
    """自动识别 v22 / v23 ckpt（见 model_v23.load_any_model）。"""
    from model_v23 import load_any_model
    model, hp, _ = load_any_model(ckpt_path, device=device, ch=ch, n_pitch=n_pitch)
    print(f'从ckpt读取hparams: {hp}', flush=True)
    return model


def predict_one(model, mel, grid, label, device='cpu', rng=None):
    """条件采音推理：50%全遮罩（盲采）+ 50%半遮罩（条件采音），和训练一致。
    返回s1(有音分数), s2(双押分数), mask_pt。阈值在外部应用，支持扫阈值。"""
    Gc = len(grid)
    if rng is None:
        rng = random.Random(42)
    if rng.random() < 0.5:
        # 全遮罩（纯盲采）
        mask_pt = 0
        notes_val = -1
    else:
        # 半遮罩（条件采音），遮罩点在2%-90%（和训练一致，支持短参考+长预测）
        mask_pt = int(Gc * rng.uniform(0.02, 0.9))
        notes_val = None  # 前半段给真实label
    with torch.no_grad():
        M = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).to(device)
        G = torch.from_numpy(grid.astype(np.float32)).to(device)
        notes = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).clone().to(device)
        if notes_val is not None:
            notes[:, :] = notes_val  # 全遮罩
        else:
            notes[:, mask_pt:] = -1  # 后半段遮罩
        o1, o2 = model(M, G, notes)
        s1 = torch.sigmoid(o1)[0].cpu().numpy()
        s2 = torch.sigmoid(o2)[0].cpu().numpy()
    return s1, s2, mask_pt


def f1_at_tol(pred_times, gt_times, tol_s):
    """复用训练脚本的双指针O(n+m) f1_inst，只取F1分数。
    原来的O(n×m)贪心匹配已废弃，reval全量评估快几十倍。"""
    return T.f1_inst(pred_times, gt_times, tol=tol_s)[0]


def eval_weight_on_val(model, va, device='cpu', max_songs=None, eval_seed=42, max_frames=30000,
                       th1=0.4, th2=0.5, scan_thresholds=False):
    """评估模型在val上的F1。
    scan_thresholds=True时，扫th1∈[0.3,0.35,0.4,0.45,0.5,0.55,0.6]，返回最优F1及对应阈值。"""
    # 先收集所有歌的预测分数（模型只跑一次，阈值只是最后过滤）
    preds = []  # (s1, s2, mask_pt, grid, gt)
    fids = va.fids
    if max_songs:
        fids = random.Random(eval_seed).sample(fids, min(max_songs, len(fids)))
    rng = random.Random(eval_seed)
    for fid in fids:
        mel, grid, label = va._load(fid)
        if mel is None: continue
        if mel.shape[0] > max_frames or len(grid) > 45000: continue
        # 随机切片：起始位置随机，长度随机（60%-100%），和训练时随机裁剪口径完全一致
        n_frames = mel.shape[0]
        if n_frames > 100:
            crop_len = int(n_frames * rng.uniform(0.6, 1.0))
            crop_start = rng.randint(0, n_frames - crop_len)
            crop_end = crop_start + crop_len
            mel = mel[crop_start:crop_end]
            t_start = crop_start / T.HR
            t_end = crop_end / T.HR
            mask = (grid >= t_start) & (grid < t_end)
            grid = grid[mask] - t_start
            label = label[mask]
        s1, s2, mask_pt = predict_one(model, mel, grid, label, device, rng=rng)
        preds.append((s1, s2, mask_pt, grid, label.astype(int)))

    def calc_f1(th1_val, th2_val):
        results = {tol: {'inst': [], 'grid': []} for tol in TOLS_S}
        dynamic_grid = []  # 动态半格距grid F1（和训练时best_grid口径一致）
        for s1, s2, mask_pt, grid, gt in preds:
            Gc = len(grid)
            cls = np.zeros(Gc, int)
            hit = s1 >= th1_val
            cls[hit] = 1
            cls[hit & (s2 >= th2_val)] = 2
            has_note_pred = (cls[mask_pt:] > 0).astype(int)
            has_note_gt = (gt[mask_pt:] > 0).astype(int)
            pred_inst = np.repeat(grid[mask_pt:], has_note_pred)
            gt_inst = np.repeat(grid[mask_pt:], has_note_gt)
            pred_grid = grid[mask_pt:][cls[mask_pt:] > 0]
            gt_grid = grid[mask_pt:][gt[mask_pt:] > 0]
            for tol in TOLS_S:
                results[tol]['inst'].append(f1_at_tol(pred_inst, gt_inst, tol))
                results[tol]['grid'].append(f1_at_tol(pred_grid, gt_grid, tol))
            # 动态半格距tol（和训练时evaluate一致：median(diff(grid))*0.5）
            if len(grid) > 1:
                step_sec = np.median(np.diff(grid))
                dyn_tol = step_sec * 0.5
                dynamic_grid.append(f1_at_tol(pred_grid, gt_grid, dyn_tol))
        summary = {}
        for tol in TOLS_S:
            summary[f'{int(tol*1000)}ms'] = {
                'inst_f1': float(np.mean(results[tol]['inst'])) if results[tol]['inst'] else None,
                'grid_f1': float(np.mean(results[tol]['grid'])) if results[tol]['grid'] else None,
            }
        summary['dynamic_grid'] = {
            'grid_f1': float(np.mean(dynamic_grid)) if dynamic_grid else None,
        }
        return summary

    if not scan_thresholds:
        return calc_f1(th1, th2)

    # 扫阈值，找最优动态半格距grid F1（和训练时best_grid口径一致）
    best_f1, best_th1, best_summary = 0.0, th1, None
    for t1 in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        s = calc_f1(t1, th2)
        f1 = s['dynamic_grid']['grid_f1']
        if f1 > best_f1:
            best_f1, best_th1, best_summary = f1, t1, s
    best_summary['best_th1'] = best_th1
    best_summary['scanned'] = True
    return best_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=None, help='数据根目录（自动搜索grid_npz_full128）')
    ap.add_argument('--data', default='grid_npz_full128')
    ap.add_argument('--split', default='split_full128_final.json')
    ap.add_argument('--pitch-dir', default='pitch_npz_full')
    ap.add_argument('--tag', default='v22')
    ap.add_argument('--ch', type=int, default=96, help='模型通道数（从ckpt自动读取）')
    ap.add_argument('--n-pitch', type=int, default=46, help='pitch特征维度')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--max-songs', type=int, default=None, help='评估歌曲数，None=全量')
    ap.add_argument('--max-frames', type=int, default=30000, help='长歌过滤帧数（和训练--max-frames一致）')
    ap.add_argument('--scan-thresholds', action='store_true', help='扫阈值选最优（th1∈0.3-0.6），默认用固定0.4/0.5')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    args = ap.parse_args()

    # 自动搜索数据目录：如果指定了data-root就用它，否则在当前目录和上级目录找grid_npz_full128
    if args.data_root:
        root = args.data_root
    else:
        root = '.'
        for _ in range(3):  # 当前目录 + 往上找2级
            if os.path.isdir(os.path.join(root, 'grid_npz_full128')):
                break
            if os.path.isdir(os.path.join(root, args.data)):
                break
            root = os.path.join(root, '..')
    args.data = os.path.join(root, args.data) if not os.path.isabs(args.data) else args.data
    args.split = os.path.join(root, args.split) if not os.path.isabs(args.split) else args.split
    args.pitch_dir = os.path.join(root, args.pitch_dir) if not os.path.isabs(args.pitch_dir) else args.pitch_dir
    print(f'数据目录: {root}', flush=True)

    print(f'reval v22: device={args.device} ch={args.ch} topk={args.topk}', flush=True)
    va = T.Songs('val', args.data, args.split,
                 pitch_dir=args.pitch_dir, pitch_mode='chroma',
                 n_pitch=args.n_pitch, flux=True)
    print(f'val {len(va)} 首', flush=True)

    cands = sorted(glob.glob(f'ckpt_{args.tag}_ep*.pt'))
    if not cands:
        # fallback：训练中途崩了只有ckpt_{tag}.pth（断点续训文件）
        fallback = f'ckpt_{args.tag}.pth'
        if os.path.exists(fallback):
            cands = [fallback]
            print(f'未找到ep快照，使用断点文件 {fallback}', flush=True)
    if not cands:
        print('没有找到 ckpt', flush=True)
        return
    print(f'找到 {len(cands)} 个 checkpoint', flush=True)

    best_f1, best_path, best_th1, all_results = 0.0, None, 0.4, []
    test_result = None  # 保存test评估结果，供后面写json用
    for path in cands:
        t0 = time.time()
        try:
            model = load_model(path, ch=args.ch, n_pitch=args.n_pitch, device=args.device)
            r = eval_weight_on_val(model, va, args.device, args.max_songs, max_frames=args.max_frames,
                                   scan_thresholds=args.scan_thresholds)
            f1 = r['dynamic_grid']['grid_f1']  # 动态半格距grid F1选冠军（和训练时best_grid口径完全一致）
            all_results.append((path, f1, r))
            if f1 is None:
                print(f'{os.path.basename(path)}: f1=无有效样本（没歌可评估，请检查数据目录） ({time.time()-t0:.1f}s)', flush=True)
                continue
            print(f'{os.path.basename(path)}: f1={f1:.4f} ({time.time()-t0:.1f}s)', flush=True)
            if f1 > best_f1:
                best_f1, best_path = f1, path
                # 保存冠军的最优阈值（如果开启了扫阈值），test评估用同一阈值，口径一致
                best_th1 = r.get('best_th1', 0.4)
        except Exception as e:
            print(f'{os.path.basename(path)}: 失败 {e}', flush=True)

    all_results.sort(key=lambda x: -(x[1] if x[1] is not None else -1.0))  # None排最后
    print(f'\n=== Top-{args.topk} ===', flush=True)
    for path, f1, r in all_results[:args.topk]:
        f1_str = f'{f1:.4f}' if f1 is not None else '无有效样本'
        print(f'{os.path.basename(path)}: f1={f1_str}', flush=True)
        print(f'  {r}', flush=True)

    if best_path:
        import shutil
        shutil.copy(best_path, f'beat_fcn_best_{args.tag}.pt')
        print(f'\n最佳: {os.path.basename(best_path)} f1={best_f1:.4f}', flush=True)
        print(f'已复制为 beat_fcn_best_{args.tag}.pt', flush=True)
        # 在独立test集上评估冠军模型（只跑一次，无偏估计）
        try:
            ta = T.Songs('test', args.data, args.split,
                         pitch_dir=args.pitch_dir, pitch_mode='chroma',
                         n_pitch=args.n_pitch, flux=True)
            if len(ta) > 0:
                print(f'\ntest {len(ta)} 首，评估冠军模型（th1={best_th1}，和选冠军同阈值）...', flush=True)
                model = load_model(best_path, ch=args.ch, n_pitch=args.n_pitch, device=args.device)
                # 修复：test评估用冠军的最优阈值，和选冠军口径一致
                # 之前固定用0.4，导致扫阈值选出来的冠军在test上分数偏低
                tr = eval_weight_on_val(model, ta, args.device, None, max_frames=args.max_frames,
                                        th1=best_th1, th2=0.5)
                print(f'=== TEST 最终分数（无偏估计，th1={best_th1}）===', flush=True)
                print(f'  dynamic_grid F1: {tr["dynamic_grid"]["grid_f1"]:.4f}  ← 和选冠军同口径', flush=True)
                print(f'  11ms inst_f1: {tr["11ms"]["inst_f1"]:.4f}', flush=True)
                print(f'  11ms grid_f1: {tr["11ms"]["grid_f1"]:.4f}', flush=True)
                print(f'  0ms inst_f1: {tr["0ms"]["inst_f1"]:.4f}', flush=True)
                test_result = {
                    "test_count": len(ta),
                    "th1": best_th1,
                    "dynamic_grid_f1": tr["dynamic_grid"]["grid_f1"],
                    "inst_f1_11ms": tr["11ms"]["inst_f1"],
                    "grid_f1_11ms": tr["11ms"]["grid_f1"],
                    "inst_f1_0ms": tr["0ms"]["inst_f1"],
                }
            else:
                print('test集为空，跳过test评估', flush=True)
        except Exception as e:
            print(f'test评估失败: {e}', flush=True)

    # 保存完整评估结果（文档承诺的reval_v22_result.json）
    result = {
        "tag": args.tag,
        "device": args.device,
        "val_count": len(va),
        "topk": args.topk,
        "scan_thresholds": args.scan_thresholds,
        "best": {
            "ckpt": os.path.basename(best_path) if best_path else None,
            "val_grid_f1": best_f1,
            "best_th1": best_th1,
        },
        "all_ckpts": [
            {"ckpt": os.path.basename(p), "val_grid_f1": f1,
             "inst_f1_11ms": r["11ms"]["inst_f1"], "grid_f1_11ms": r["11ms"]["grid_f1"]}
            for p, f1, r in all_results
        ],
        "test": test_result,
    }
    with open(f"reval_v22_result_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存到 reval_v22_result_{args.tag}.json", flush=True)


if __name__ == '__main__':
    main()
