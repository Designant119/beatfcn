#!/usr/bin/env python3
"""v22 vs v23 单曲 前向+反向 速度 / FLOPs / 峰值内存 基准（每个配置单独子进程，测 ru_maxrss）。
用法: python bench/bench_speed.py [--device cpu] [--frames 3000 6000 11000]"""
import sys, os, json, time, argparse, subprocess, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_one(arch, T, device, reps):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    torch.manual_seed(0)
    G = int(T / 86.13 * 2 * 32)  # 120BPM：每秒 2 拍 × 32 格
    G = int(T * 0.79)  # ≈ 实际数据 (Crisscross 11198 帧 / 8865 格)
    if arch == 'v22':
        from model import BeatFCNContext
        m = BeatFCNContext(n_mels=81, ch=96, n_pitch=46)
    else:
        from model_v23 import BeatTokenNet
        m = BeatTokenNet()
    m.to(device).train()
    mel = torch.randn(1, T, 127, device=device)
    grid = (torch.arange(G, device=device).float() * (T / 86.13) / G).unsqueeze(0)
    notes = torch.full((1, G), -1.0, device=device)
    y = (torch.rand(1, G, device=device) < 0.06).float()
    def step():
        o1, o2 = m(mel, grid, notes)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(o1, y) + o2.mean() * 0
        loss.backward()
    with FlopCounterMode(display=False) as fc:
        step()
    flops = fc.get_total_flops()
    m.zero_grad()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); step(); m.zero_grad(); ts.append(time.perf_counter() - t)
    m.eval()
    with torch.no_grad():
        t = time.perf_counter(); m(mel, grid, notes); tinf = time.perf_counter() - t
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    return dict(arch=arch, T=T, G=G, train_gflops=flops / 1e9, train_s=min(ts), infer_s=tinf, peak_rss_mb=rss)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--frames', type=int, nargs='+', default=[3000, 6000, 11000])
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--one', nargs=2)
    a = ap.parse_args()
    if a.one:
        import torch; torch.set_num_threads(a.threads)
        print(json.dumps(run_one(a.one[0], int(a.one[1]), a.device, a.reps)))
        sys.exit()
    rows = []
    for T in a.frames:
        for arch in ['v22', 'v23']:
            r = subprocess.run([sys.executable, __file__, '--one', arch, str(T), '--device', a.device,
                                '--reps', str(a.reps), '--threads', str(a.threads)], capture_output=True, text=True)
            if r.returncode != 0:
                row = dict(arch=arch, T=T, error=(r.stderr.strip().splitlines() or ['killed (OOM?)'])[-1][:120])
            else:
                row = json.loads(r.stdout.strip().splitlines()[-1])
            print(json.dumps(row, ensure_ascii=False), flush=True)
            rows.append(row)
    json.dump(rows, open('bench_speed_result.json', 'w'), indent=1)
