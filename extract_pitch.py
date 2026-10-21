#!/usr/bin/env python3
"""从 .pez 谱面批量提取 pitch/打击特征（46 维）
与 grid_npz_v13 的 mel 帧对齐：SR=22050, N_FFT=1024, HOP=256
纯 numpy + soundfile + scipy（不用 librosa；HPSS 用 numpy 中值滤波实现）

输出特征（46 维，全部与 mel 同帧定义，中心对齐窗口）：
  chroma_n  12  半音级能量占比（L1 归一化）
  oct_log    8  八度能量 log1p
  rms        1  帧能量 RMS log1p（Parseval 从功率谱算）
  flux       1  频谱通量（帧间幅度 L1 差）log1p
  subband    4  四子带能量 log1p（20-200/200-800/800-3000/3000-8000 Hz）
  hpss_mel 20  打击分量 mel 能量 log1p（时间中值滤波分离谐波）

输出: {out_dir}/{id}.npz  ->  {"pitch": (T,46) float32, "frame_n": int}
"""
import os, sys, glob, zipfile, io, json
import numpy as np
import signal
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chart_utils import SR, N_FFT, HOP, _stft_power, decode_audio

N_CHROMA = 12
N_OCT = 8            # C1(32.7Hz) ~ B7
N_SUB = 4            # 四子带
N_HPSS_MEL = 20      # 打击分量 mel 带数
EPS = 1e-7

# 音级基准频率（A440，k=9 对应 440Hz；k=0=C）
_CHROMA_FREQ = np.array([440.0 * 2 ** ((k - 9) / 12.0) for k in range(12)])
# 八度边界（C1=32.70Hz 起，每个八度 ×2）
_OCT_EDGES = 32.70 * 2 ** np.arange(N_OCT + 1)
# 四子带边界（Hz）
_SUB_EDGES = np.array([20.0, 200.0, 800.0, 3000.0, 8000.0])

# 预计算 FFT bin 频率
_freqs = None
def _bin_freqs():
    global _freqs
    if _freqs is None:
        _freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    return _freqs

# 预计算 chroma 映射权重（bins × 12）
_chroma_map = None
def _get_chroma_map():
    global _chroma_map
    if _chroma_map is not None:
        return _chroma_map
    freqs = _bin_freqs()
    cents = 1200.0 * np.log2(np.maximum(freqs, 1e-3) / 440.0)  # 音分
    cents_mod = np.mod(cents, 1200.0)                           # 一个八度内位置
    idx = (cents_mod / 100.0).astype(int) % 12                  # 最近音级
    frac = cents_mod / 100.0 - np.floor(cents_mod / 100.0)      # 距下音级距离
    w = np.zeros((len(freqs), 12), dtype=np.float32)
    w[np.arange(len(freqs)), idx] += 1.0 - frac
    w[np.arange(len(freqs)), (idx + 1) % 12] += frac
    # 极低频(<C1/2)丢弃：frequency 太低无音高意义
    low = freqs < 30.0
    w[low, :] = 0.0
    _chroma_map = w
    return w

# 预计算八度归属（bins × N_OCT）
_oct_map = None
def _get_oct_map():
    global _oct_map
    if _oct_map is not None:
        return _oct_map
    freqs = _bin_freqs()
    w = np.zeros((len(freqs), N_OCT), dtype=np.float32)
    for i in range(N_OCT):
        lo, hi = _OCT_EDGES[i], _OCT_EDGES[i + 1]
        m = (freqs >= lo) & (freqs < hi)
        w[m, i] = 1.0
    # 超出范围（<C1 或 >=B7 顶端）：归入最近八度
    w[freqs < _OCT_EDGES[0], 0] = 1.0
    w[freqs >= _OCT_EDGES[-1], -1] = 1.0
    _oct_map = w
    return w

# 预计算子带归属（bins × N_SUB）
_sub_map = None
def _get_sub_map():
    global _sub_map
    if _sub_map is not None:
        return _sub_map
    freqs = _bin_freqs()
    w = np.zeros((len(freqs), N_SUB), dtype=np.float32)
    for i in range(N_SUB):
        lo, hi = _SUB_EDGES[i], _SUB_EDGES[i + 1]
        m = (freqs >= lo) & (freqs < hi)
        w[m, i] = 1.0
    w[freqs < _SUB_EDGES[0], 0] = 1.0            # 超低频归入第一带
    w[freqs >= _SUB_EDGES[-1], -1] = 1.0         # 超高频归入最后一带
    _sub_map = w
    return w

# 预计算 20 带 mel 三角滤波器（bins × N_HPSS_MEL）——用于打击分量
_mel_map = None
def _get_mel_map():
    global _mel_map
    if _mel_map is not None:
        return _mel_map
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    fmin, fmax = 20.0, 8000.0
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), N_HPSS_MEL + 2)
    hzs = mel2hz(mels)
    freqs = _bin_freqs()
    bins = np.floor((N_FFT + 1) * hzs / SR).astype(int)
    w = np.zeros((len(freqs), N_HPSS_MEL), dtype=np.float32)
    for i in range(N_HPSS_MEL):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            w[l:c, i] = (np.arange(l, c) - l) / max(c - l, 1)
        if r > c:
            w[c:r, i] = (r - np.arange(c, r)) / max(r - c, 1)
    _mel_map = w
    return w

def extract_pitch(wav):
    """输入 wav -> (T, 46) pitch 特征（顺序：chroma12, oct8, rms1, flux1, sub4, hpss_mel20）"""
    spec = _stft_power(wav)
    if spec is None:
        return None
    cmap = _get_chroma_map()   # (bins, 12)
    omap = _get_oct_map()      # (bins, 8)
    smap = _get_sub_map()      # (bins, 4)
    mmap = _get_mel_map()      # (bins, 20)
    chroma = spec @ cmap       # (T, 12)
    oct_en = spec @ omap       # (T, 8)
    sub_en = spec @ smap       # (T, 4)
    # L1 归一化 chroma（帧能量足够时），静音帧置 0
    rs = chroma.sum(axis=1, keepdims=True)
    chroma_n = np.where(rs > EPS, chroma / np.maximum(rs, EPS), 0.0).astype(np.float32)
    oct_log = np.log1p(oct_en).astype(np.float32)
    sub_log = np.log1p(sub_en).astype(np.float32)
    # RMS：Parseval 定理 时域能量 = Σ|X[k]|²/N_FFT，rms = sqrt(2*Σspec/N_FFT)（单边谱×2）
    energy = spec.sum(axis=1)
    rms = np.sqrt(np.maximum(2.0 * energy / N_FFT, 0.0))
    rms_log = np.log1p(rms).astype(np.float32)[:, None]
    # 频谱通量：帧间幅度 L1 差（log1p 压缩）
    amp = np.sqrt(np.maximum(spec, 0.0))
    flux = np.zeros(amp.shape[0], dtype=np.float32)
    if amp.shape[0] > 1:
        flux[1:] = np.abs(amp[1:] - amp[:-1]).sum(axis=1)
    flux_log = np.log1p(flux).astype(np.float32)[:, None]
    # HPSS：沿时间轴中值滤波（核 7 帧）得谐波，打击 = 幅度 - 谐波（clip≥0）
    harm = median_filter(amp, size=(7, 1), mode='reflect')
    perc = np.maximum(amp - harm, 0.0)
    hpss_mel = np.log1p(perc ** 2 @ mmap).astype(np.float32)
    # 拼接 46 维
    pitch = np.concatenate([chroma_n, oct_log, rms_log, flux_log, sub_log, hpss_mel], axis=1)
    return pitch.astype(np.float32)

def process(pez_path, out_path):
    try:
        z = zipfile.ZipFile(pez_path)
        an = next((n for n in z.namelist() if n.lower().endswith(('.ogg', '.mp3', '.flac', '.wav'))), None)
        if not an:
            z.close()
            return 'no_audio'
        y = decode_audio(z.read(an))
        z.close()
        if y is None or len(y) < SR:
            return 'decode_fail'
        if len(y) > SR * 360:  # 超过6分钟的歌跳过，和extract_full.py一致
            return 'too_long'
        pitch = extract_pitch(y)
        if pitch is None or pitch.shape[0] < 10:
            return 'too_short'
        if pitch.shape[1] != 46:
            return f'err:dims_{pitch.shape[1]}'
        # 原子写入：先写临时文件，再rename，避免进程被杀留下半写的损坏npz
        tmp_path = out_path + '.tmp.npz'  # savez自动加.npz，这里手动加好避免后缀错位
        np.savez(tmp_path, pitch=pitch.astype(np.float32), frame_n=pitch.shape[0])
        os.replace(tmp_path, out_path)
        return 'ok'
    except Exception as e:
        return f'err:{e}'

def run_with_timeout(fn, *args, timeout=90):
    """单文件处理带超时（HPSS 慢，放宽到 90 秒）：损坏音频卡死时返回 'timeout'"""
    def handler(sig, frm):
        raise TimeoutError('file_timeout')
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args)
    except TimeoutError:
        return 'timeout'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

if __name__ == '__main__':
    from multiprocessing import Pool, cpu_count
    pez_dir, out_dir = sys.argv[1], sys.argv[2]
    id_file = sys.argv[3] if len(sys.argv) > 3 else None  # 只提取该文件里的 id（每行一个）
    nprocs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    # 自动降级：CPU核数不够就降到实际核数
    _cpus = cpu_count()
    if nprocs > _cpus:
        print(f"[自动降级] 请求{nprocs}进程，但CPU只有{_cpus}核，降为{_cpus}进程", flush=True)
        nprocs = _cpus
    if _cpus <= 1:
        print(f"[警告] 当前CPU只有{_cpus}核，单进程提取会非常慢。建议在8核以上机器跑。", flush=True)
    want = None
    if id_file:
        want = set(l.strip() for l in open(id_file) if l.strip())
        print(f'过滤: 只提取 {len(want)} 个 id', flush=True)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(f'{pez_dir}/**/*.zip', recursive=True))
    if want:
        files = [f for f in files if os.path.basename(f)[:-4] in want]
    print(f'共{len(files)}个zip，{nprocs}进程并行', flush=True)
    jobs = [(f, f'{out_dir}/{os.path.basename(f)[:-4]}.npz') for f in files]
    def _worker(args):
        pf, out = args
        if os.path.exists(out):
            return 'skip'
        return run_with_timeout(process, pf, out)
    ok = bad = skip = 0
    with Pool(nprocs) as pool:
        for r in pool.imap_unordered(_worker, jobs, chunksize=8):
            if r == 'ok':
                ok += 1
            elif r == 'skip':
                skip += 1
            else:
                bad += 1
            if (ok + bad + skip) % 100 == 0 and ok + bad + skip > 0:
                print(f'ok{ok} bad{bad} skip{skip}', flush=True)
    print(f'完成! ok={ok} bad={bad} skip={skip}', flush=True)
