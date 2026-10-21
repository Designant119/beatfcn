#!/usr/bin/env python3
"""全量 zip 谱面集 → grid/pitch npz。
用法: python3 extract_full.py <zip_dir> <out_dir> <mode:grid|pitch> <nprocs> [id_file] [div]
mode=grid : 输出 {id}.npz (mel(T,80), grid, label 0/1/2)；div=每拍格数（16分=4, 32分=8, 128分=32，默认32）
mode=pitch: 输出 {id}.npz (pitch(T,46), frame_n)
"""
import sys, os, glob, zipfile, gc, time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 脚本同目录
from chart_utils import parse_chart, parse_pec, chart_sec, beat_to_sec, beats_of, beat_grid, mel_of, SR, HR, TOL, decode_audio
from extract_pitch import extract_pitch


MODE = sys.argv[3] if len(sys.argv) > 3 else 'grid'
NPROCS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ID_FILE = sys.argv[5] if len(sys.argv) > 5 else None
DIV = int(sys.argv[6]) if len(sys.argv) > 6 else 32   # 每拍格数：16分=4, 32分=8, 64分=16, 128分=32（默认32=128分）
STEP = 1.0 / DIV

# 自动降级：如果CPU核数少于请求的进程数，自动降到实际核数
_cpus = cpu_count()
if NPROCS > _cpus:
    print(f"[自动降级] 请求{NPROCS}进程，但CPU只有{_cpus}核，降为{_cpus}进程", flush=True)
    NPROCS = _cpus
if _cpus <= 1:
    print(f"[警告] 当前CPU只有{_cpus}核，单进程提取会非常慢（13876个谱面可能需要20+小时）。", flush=True)
    print(f"       建议：在8核以上的机器上跑（如AutoDL GPU实例，或本地电脑）。", flush=True)


def load_zip(zip_path):
    """解压 zip，返回 (json_list[(name, bytes)], pec_list[(name, bytes)], audio_bytes, info_offset秒, difficulty, song_name, level_str)。
    info_offset从info.yml读取（RPE谱面offset单位秒），PEC文本谱面的offset在parse_pec里读。
    difficulty从info.yml读取，用于过滤troll/visual谱面。
    song_name从info.yml的name字段读取，用于生成可读的文件名。
    level_str从info.yml的level字段读取（如"AT Lv.15"），用于生成难度标签。
    json_list保留文件名，用于区分EZ/HD/IN/AT多难度。"""
    try:
        z = zipfile.ZipFile(zip_path)
        jbs, pbs, an = [], [], None
        info_offset = 0.0
        difficulty = None
        song_name = ''
        level_str = ''
        for n in z.namelist():
            low = n.lower()
            if low.endswith('.json') and 'info' not in n and 'extra' not in n and '封面' not in n and '.jpg' not in n and '.png' not in n and 'autosave' not in low:
                jbs.append((n, z.read(n)))  # 保留文件名
            elif low.endswith('.pec'):
                pbs.append((n, z.read(n)))
            elif low.endswith(('.ogg', '.mp3', '.flac', '.wav')):
                if an is None:
                    an = n
            elif low.endswith('info.yml') or low.endswith('info.yaml'):
                # RPE谱面的offset和difficulty写在info.yml里
                try:
                    info_txt = z.read(n).decode('utf-8-sig', errors='replace')
                    for line in info_txt.split('\n'):
                        line = line.strip()
                        low_line = line.lower()
                        if low_line.startswith('offset'):
                            val = line.split(':', 1)[-1].split('=', 1)[-1].strip()
                            info_offset = float(val)
                        elif low_line.startswith('difficulty'):
                            val = line.split(':', 1)[-1].split('=', 1)[-1].strip()
                            try:
                                difficulty = float(val)
                            except ValueError:
                                difficulty = None  # 非数字难度（如'MKTT'）视为troll
                        elif low_line.startswith('level') and not low_line.startswith('levellist'):
                            val = line.split(':', 1)[-1].split('=', 1)[-1].strip()
                            if val:
                                level_str = val
                        elif low_line.startswith('name') and not low_line.startswith('namelist'):
                            val = line.split(':', 1)[-1].split('=', 1)[-1].strip()
                            if val:
                                song_name = val
                except Exception:
                    pass
        if (not jbs and not pbs) or an is None:
            z.close(); return None
        ab = z.read(an)
        z.close(); return jbs, pbs, ab, info_offset, difficulty, song_name, level_str
    except Exception:
        return None


def _parse_level_tag(level_str, fallback_filename=''):
    """从info.yml的level字段解析难度标签，完整保留信息不省略。
    如"AT Lv.15"→"[AT15]"，"SP Lv.??"→"[SP??]"，"SP Lv.Yiep Verse"→"[SPYiepVerse]"。
    解析失败时从json文件名推断，再失败用level原文。"""
    import re
    if not level_str:
        return _diff_name_from_filename(fallback_filename)
    level_clean = level_str.strip().strip("'\"")  # 去掉引号
    # 1. 难度名+Lv.+后缀（后缀可以是数字、??、中文、英文短语等，完整保留到结尾）
    m = re.match(r'^\s*([A-Za-z]+)\s*Lv\.?\s*(.+)$', level_clean, re.IGNORECASE)
    if m:
        diff_name = m.group(1).upper()
        suffix = re.sub(r'[\s\\/:*"<>|]', '', m.group(2))  # 去掉空格和非法字符，保留?等符号
        return f'[{diff_name}{suffix}]'
    # 2. 难度名+数字直接写：AT15 / IN14 / HD10
    m = re.match(r'^\s*([A-Za-z]+)\s*(\d+)\s*$', level_clean)
    if m:
        return f'[{m.group(1).upper()}{m.group(2)}]'
    # 3. 难度名+非数字后缀：SP? / SP?? / SP双指 → [SP?] / [SP??] / [SP双指]
    m = re.match(r'^\s*([A-Za-z]+)\s*(\S+)\s*$', level_clean)
    if m:
        return f'[{m.group(1).upper()}{m.group(2)}]'
    # 4. 只有难度名：SP / AT / Speed → [SP] / [AT] / [SPEED]
    m = re.match(r'^\s*([A-Za-z]+)\s*$', level_clean)
    if m:
        return f'[{m.group(1).upper()}]'
    # 5. 只有数字：'14' / 15 → [14]
    m = re.search(r'(\d+)', level_clean)
    if m:
        return f'[{m.group(1)}]'
    # 6. level非空但完全匹配不到，用level原文（清洗后前15字符，不省略）
    clean = re.sub(r'[\s\\/:*"<>|]', '', level_clean)[:15]
    if clean:
        return f'[{clean}]'
    # 从json文件名推断
    return _diff_name_from_filename(fallback_filename)


def _diff_name_from_filename(fname):
    """从json文件名推断难度标签，返回可读格式如 [AT15]、[HD10]、[EZ5]。
    RPE谱面常见命名：AT15.json / IN14.json / HD10.json / EZ5.json，或 0.json/1.json/2.json/3.json"""
    import re
    base_orig = os.path.basename(fname).replace('.json', '').replace('.pec', '')
    base = base_orig.lower()
    # 匹配 难度字母+数字，如 at15、in14、hd10、ez5
    m = re.match(r'^(at|in|hd|ez)[\s_\-]*(\d+)$', base)
    if m:
        return f'[{m.group(1).upper()}{m.group(2)}]'
    # 匹配纯数字：0=EZ, 1=HD, 2=IN, 3=AT（Phigros惯例）
    try:
        idx = int(base)
        names = ['EZ', 'HD', 'IN', 'AT']
        if idx < 4:
            return f'[{names[idx]}]'
        return f'[D{idx}]'
    except ValueError:
        pass
    # 兜底：用文件名前8字符
    clean = re.sub(r'[^\w]', '', base_orig)[:8]
    return f'[{clean}]' if clean else '[chart]'


def _tag_from_meta_level(meta_level, fallback_filename=''):
    """从json META.level字段（难度索引0=EZ,1=HD,2=IN,3=AT）生成难度标签。
    META.level可能是字符串"0"/"1"/"2"/"3"，也可能是空或其他值。"""
    level_map = {0: 'EZ', 1: 'HD', 2: 'IN', 3: 'AT'}
    try:
        idx = int(meta_level)
        if idx in level_map:
            return f'[{level_map[idx]}]'
    except (ValueError, TypeError):
        pass
    # 兜底：从文件名推断
    return _diff_name_from_filename(fallback_filename)


def _safe_filename(name):
    """清洗曲名，去掉文件名非法字符（Windows兼容）"""
    import re
    # 去掉 \ / : * ? " < > | 等所有Windows非法字符，空格替换成_
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    name = name.replace(' ', '_').replace('\t', '_')
    return name[:50]  # 限制长度


def process_one(args):
    zip_path, out_base = args  # out_base是不带后缀的基础路径，如 out_dir/12345
    zip_id = os.path.basename(zip_path)[:-4]
    # 注意：不做外层"有一个npz就跳过整个zip"的判断
    # 内层循环逐个难度检查 os.path.exists(out_path)，缺哪个补哪个
    # 这样上次跑到一半崩了（EZ成功、HD失败），重跑时会自动补HD/IN/AT
    try:
        got = load_zip(zip_path)
        if got is None:
            return 'no_files'
        jbs, pbs, ab, info_offset, difficulty, song_name, level_str = got
        # 不过滤troll/visual谱面：用户要求用上所有谱面数据
        # （difficulty不在1-19范围或非数字的谱面也保留，不跳过）
        safe_name = _safe_filename(song_name) if song_name else ''
        if MODE == 'grid':
            # 收集所有可解析的谱面（json + pec），每个难度分别提取
            # 同时从json的META字段读name和level（难度索引0=EZ,1=HD,2=IN,3=AT）
            charts = []  # [(fname, chart, meta_name, meta_level_idx)]
            for fname, jb in jbs:
                txt = jb.decode('utf-8-sig', errors='replace')
                chart = parse_chart(txt)
                if not chart:
                    chart = parse_pec(txt)
                if chart:
                    meta = chart.get('META', {}) if isinstance(chart, dict) else {}
                    meta_name = meta.get('name', '')
                    meta_level = meta.get('level', '')
                    charts.append((fname, chart, meta_name, meta_level))
            if not charts and pbs:  # 无 .json 谱面 → 尝试 .pec 文本谱面
                for fname, pb in pbs:
                    chart = parse_pec(pb.decode('utf-8-sig', errors='replace'))
                    if chart:
                        meta = chart.get('META', {}) if isinstance(chart, dict) else {}
                        meta_name = meta.get('name', '')
                        meta_level = meta.get('level', '')
                        charts.append((fname, chart, meta_name, meta_level))
            if not charts:
                return 'no_chart'
            y = decode_audio(ab)
            if y is None or len(y) < SR:
                return 'decode_fail'
            mel = mel_of(y).astype(np.float32)
            T = mel.shape[0]
            # 长歌过滤：超过360秒（6分钟）的歌直接跳过，和pitch提取一致，省CPU和磁盘
            # 训练时max_frames=30000≈348秒，超长歌也用不上
            if T > int(360 * SR / HOP):
                return 'too_long'
            mel80 = mel.astype(np.float32)
            n_extracted = 0
            dens_list = []
            # 单难度谱面：用info.yml的level（更准确）；多难度：每个json用自己的META.level
            use_info_level = len(charts) == 1 and level_str
            for fname, chart, meta_name, meta_level in charts:
                # 曲名：优先info.yml的song_name（更完整），然后json META.name
                raw_name = song_name if song_name else meta_name
                safe_name = _safe_filename(raw_name) if raw_name else ''
                # 难度标签：优先info.yml level（单难度，最准确），然后json META.level索引，然后文件名推断
                if use_info_level:
                    diff_tag = _parse_level_tag(level_str, fname)
                else:
                    diff_tag = _tag_from_meta_level(meta_level, fname)
                # 只提取IN/AT难度，滤掉EZ/HD（太简单，训练价值低）
                if diff_tag.startswith('[EZ') or diff_tag.startswith('[HD'):
                    continue
                # 文件名格式：{ID}_{曲名}_{[难度]}.npz，如 1000_アイドル_[AT15].npz
                name_part = f'_{safe_name}' if safe_name else ''
                out_path = f'{out_base}{name_part}_{diff_tag}.npz'
                if os.path.exists(out_path):
                    n_extracted += 1
                    continue  # 这个难度已提取，跳过
                # offset：PEC文本谱面在META.offset（单位ms），RPE JSON谱面在info.yml（单位秒）
                # 按来源硬编码单位，不靠数值大小猜
                meta_offset = chart.get('META', {}).get('offset')
                if meta_offset is not None:
                    offset = float(meta_offset) / 1000.0  # PEC格式固定是ms，直接转秒
                else:
                    offset = info_offset  # RPE格式固定是秒
                segs = chart_sec(chart)
                grid = beat_grid(chart, segs, step=STEP).astype(np.float32)
                grid = grid + offset
                taps = []
                for jl in chart.get('judgeLineList', []):
                    for n in jl.get('notes', []):
                        if n.get('isFake'): continue
                        t = n.get('type')
                        if t not in (1, 2, 3): continue
                        taps.append(beat_to_sec(beats_of(n['startTime']), segs) + offset)
                taps = np.array(taps, dtype=np.float32)
                # 过滤负 grid（负 offset 谱面在音频开始前产生负时间网格）
                grid = grid[(grid >= 0) & (grid < T / HR)]
                taps = taps[taps >= 0]
                label = np.zeros(len(grid), dtype=np.float32)
                if len(taps):
                    step_sec = float(np.median(np.diff(grid))) if len(grid) > 1 else TOL
                    tol = min(TOL, step_sec * 0.5)
                    for t in taps:
                        ix = np.searchsorted(grid, t)
                        best, bd = -1, tol
                        for j in (ix - 1, ix):
                            if 0 <= j < len(grid):
                                d = abs(grid[j] - t)
                                if d <= bd:
                                    best, bd = j, d
                        if best >= 0:
                            label[best] = min(2, label[best] + 1)
                # 原子写入
                _tmp = out_path + f'.{os.getpid()}.tmp'
                np.savez(_tmp, mel=mel80, grid=grid.astype(np.float32), label=label)
                _tmp_f = _tmp + '.npz'
                for _ in range(30):
                    if os.path.exists(_tmp_f):
                        try:
                            os.replace(_tmp_f, out_path); break
                        except OSError: time.sleep(0.2)
                    time.sleep(0.2)
                else:
                    continue  # 这个难度写入失败，跳过
                n_extracted += 1
                dens = float((label > 0).sum()) / max(len(label), 1)
                dens_list.append(dens)
            if n_extracted == 0:
                return 'all_fail'
            avg_dens = sum(dens_list) / len(dens_list) if dens_list else 0
            return f'ok:{n_extracted}_diff:dens={avg_dens:.4f}'
        else:  # pitch（按歌曲提取，不分难度）
            out_path = out_base + '.npz'
            if os.path.exists(out_path):
                return 'skip'
            y = decode_audio(ab)
            if y is None or len(y) < SR:
                return 'decode_fail'
            if len(y) > SR * 360:
                return 'too_long'
            p = extract_pitch(y)
            if p is None or len(p) == 0:
                return 'pitch_fail'
            _tmp = out_path + f'.{os.getpid()}.tmp'
            np.savez(_tmp, pitch=p.astype(np.float32), frame_n=p.shape[0])
            _tmp_f = _tmp + '.npz'
            for _ in range(30):
                if os.path.exists(_tmp_f):
                    try:
                        os.replace(_tmp_f, out_path); break
                    except OSError: time.sleep(0.2)
                time.sleep(0.2)
            else:
                return 'err:tmp_replace'
            return 'ok'
    except Exception as e:
        return f'err:{type(e).__name__}:{e}'


def main():
    zip_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(f'{zip_dir}/**/*.zip', recursive=True))
    if ID_FILE:
        want = set(l.strip() for l in open(ID_FILE) if l.strip())
        files = [f for f in files if os.path.basename(f)[:-4] in want]
    print(f'[{MODE}] 待处理 {len(files)} 个 zip', flush=True)
    jobs = [(f, f'{out_dir}/{os.path.basename(f)[:-4]}') for f in files]  # out_base不带后缀
    ok = bad = skip = 0
    total_diff = 0
    from collections import Counter
    badc = Counter()
    dense_songs = []  # 异常密度的歌（>30%或<0.5%）
    with Pool(NPROCS) as pool:
        for i, r in enumerate(pool.imap_unordered(process_one, jobs, chunksize=8)):
            if r.startswith('ok'):
                ok += 1
                # 解析：ok:N_diff:dens=X （grid模式多难度）
                if '_diff:' in r:
                    n_diff = int(r.split(':')[1].replace('_diff', ''))
                    total_diff += n_diff
                if ':dens=' in r:
                    dens = float(r.split('=')[1])
                    if dens > 0.30 or dens < 0.005:
                        dense_songs.append((os.path.basename(jobs[i][0]), dens))
            elif r == 'skip':
                skip += 1
            else:
                bad += 1; badc[r] += 1
            if (ok + bad + skip) % 300 == 0 and ok + bad + skip > 0:
                print(f'[{MODE}] ok={ok} bad={bad} skip={skip} 谱面数={total_diff} 失败: {dict(badc)}', flush=True)
    print(f'[{MODE}] 完成! ok={ok} bad={bad} skip={skip} 谱面总数={total_diff} 失败明细: {dict(badc)}', flush=True)
    # 数据质量报告：异常密度的歌
    if dense_songs:
        dense_songs.sort(key=lambda x: -x[1])
        print(f'\n=== 数据质量报告：异常密度歌 {len(dense_songs)} 首（>30%或<0.5%）===', flush=True)
        for name, dens in dense_songs[:20]:
            print(f'  {name}: 有音格占比={dens*100:.1f}%', flush=True)
        if len(dense_songs) > 20:
            print(f'  ... 还有 {len(dense_songs)-20} 首', flush=True)


if __name__ == '__main__':
    main()
