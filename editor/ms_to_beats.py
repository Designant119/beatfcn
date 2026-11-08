"""毫秒(模型钟) -> Phira 拍分数 [整拍, 分子, 分母] 的精确换算。

前提:制谱器提供权威 BPM 图(每段变速的 BPM),与仓库内 beatfcn/ingest/timing.py 同一约定:
时间在谱面自己的节拍网格上是精确分数。换算是确定性的,不是吸附。

设计要点(2026-11-08 实测定稿,教训写在对话记录里):
- 排序键 = 误差 + 分母复杂度罚分(拍长×分母×系数),音乐族只当并列 tie-break:
  纯"族优先"会让 3/64 偷走 1/20、9/28 偷走 7/22;纯"最小分母"会把 1/16 偷换成 1/14;
  纯"最小误差"会跑飞到 3/47 或被大分母的数值巧合淹没。
- 容差:min(2ms, 1/4格)。相邻怪格族(如 1/22 与 1/24 在 174 BPM 只差 1.3ms)天然无法
  用毫秒唯一判定——所以本函数永远返回按可信度排序的候选列表,编辑器亮前几个给谱师一键选。
- 分辨极限是物理事实:任何毫秒位置总在某个 d<=64 的分数 2ms 内,"完全不贴格"不存在。
纯标准库,无 torch;自测:python editor/ms_to_beats.py
"""
from bisect import bisect_right
from math import gcd

MAX_D = 64


def _musical_denominators(max_d=MAX_D):
    """音乐常用度优先的分母顺序:偶数分族先,三连次之,五连、七连……最后杂项升序。"""
    families = [
        [1, 2, 4, 8, 16, 32, 64],          # 二等分(最常见)
        [3, 6, 12, 24, 48],                # 三连族
        [5, 10, 20, 40],                   # 五连族
        [7, 14, 28],                       # 七连族
        [9, 18, 36], [11, 22, 44], [13, 26, 52], [15, 30, 60],
    ]
    seen, ordered = set(), []
    for fam in families:
        for d in fam:
            if d <= max_d and d not in seen:
                seen.add(d)
                ordered.append(d)
    ordered += [d for d in range(1, max_d + 1) if d not in seen]
    return ordered


MUSICAL_DENS = _musical_denominators()
_FAMILY_RANK = {d: k for k, d in enumerate(MUSICAL_DENS)}
# 族类别(供 sure 判定):0=二等分 1=三连 2=五连 3=七连 4=其他/九连以上 —— 类别<3 视为常见格
_FAMILY_CLASS = {}
for _cls, _fam in enumerate(([1, 2, 4, 8, 16, 32, 64], [3, 6, 12, 24, 48],
                             [5, 10, 20, 40], [7, 14, 28], [9, 18, 36], [11, 22, 44])):
    for _d in _fam:
        _FAMILY_CLASS[_d] = _cls


def _family_rank(d):
    """并列误差时,标准格族(二等分->三连->五连...)排前面;见模块注释的偷换教训。"""
    return _FAMILY_RANK.get(d, 99)


def _family_class(d):
    return _FAMILY_CLASS.get(d, 6)


def beat_value(whole, num, den):
    """Phira startTime [整拍, 分子, 分母] -> 浮点拍位置。"""
    if den == 0:
        raise ValueError('denominator must not be 0')
    return whole + num / den


def parse_bpm_list(bpm_list):
    """Phira BPMList [{bpm, startTime:[b,n,d]}] -> [(起始拍, bpm)] 升序;同拍多点只留最后一个。"""
    segs = sorted((beat_value(*e['startTime']), float(e['bpm'])) for e in bpm_list)
    out = []
    for b, bpm in segs:
        if out and abs(b - out[-1][0]) < 1e-9:
            out[-1] = (out[-1][0], bpm)
        else:
            out.append((b, bpm))
    if not out:
        raise ValueError('empty BPM list')
    return out


def segment_times(segs):
    """每段起始拍对应的绝对秒(从 0 起,按各段 BPM 积分)。"""
    times = [0.0]
    for i in range(1, len(segs)):
        beats = segs[i][0] - segs[i - 1][0]
        times.append(times[-1] + beats * 60.0 / segs[i - 1][1])
    return times


def time_to_beat(t_us, segs, times=None):
    """微秒(模型钟,歌曲起点为 0)-> 浮点拍位置(含变速段的分段换算)。"""
    times = times if times is not None else segment_times(segs)
    t = t_us / 1e6
    i = max(0, min(bisect_right(times, t) - 1, len(segs) - 1))
    return segs[i][0] + (t - times[i]) * segs[i][1] / 60.0


def beat_ms_of(segs, times, t_us):
    """事件所在段的拍长(毫秒),用于容差缩放。"""
    i = max(0, min(bisect_right(times, t_us / 1e6) - 1, len(segs) - 1))
    return 60_000.0 / segs[i][1]


def beat_fraction(x, beat_ms, max_d=MAX_D, top=8, allowed_dens=None):
    """浮点拍 -> 候选列表 [(整拍,分子,分母,误差ms), ...] 按可信度排序,最多 top 个。

    allowed_dens:编辑器吸附菜单里的分母集合(如 [1,2,4,8,16,3,6,12,24,...])。传入后
    候选只在这些格子里找——数值巧合(如 1/15 抢 1/16)被彻底排除,小误差时真值必排第一。
    不传则全分母搜索,只承诺"真值在候选列表里",由谱师在候选中一键确认。
    """
    dens = sorted(allowed_dens or range(1, max_d + 1))
    cands = {}
    for d in dens:
        if not 1 <= d <= max_d:
            continue
        slot = beat_ms / d
        n = round(x * d)
        err_ms = abs(x - n / d) * beat_ms
        if err_ms < min(5.0, slot * 0.5):  # 候选从宽:真格子可能被模型误差推出 2ms,必须留在列表里
            whole, num = divmod(n, d)
            if num:
                g = gcd(num, d)
                num, d = num // g, d // g
            else:
                num, d = 0, 1
            # 罚分随(约分前的)分母线性增长:大分母的数值巧合(误差小但没音乐意义)排不到前面去;
            # 同一约分结果(如 1/14 = 2/28 = 3/42)只留分母最小的那条,否则重复候选淹没列表
            score = err_ms + beat_ms * d * 4.35e-4
            key = (whole, num, d)
            if key not in cands or score < cands[key][0]:
                cands[key] = (score, err_ms, d, whole, num)
    ordered = sorted(cands.values(), key=lambda c: (round(c[0], 3), _family_rank(c[2]), c[2]))
    return [(w, n, d, e) for _, e, d, w, n in ordered[:top]]


def convert_event(t_us, segs, times, allowed_dens=None):
    """单个时间戳 -> dict(首选[b,n,d] + 候选列表);永不"失败",拿不准时靠候选亮给谱师。"""
    beat_ms = beat_ms_of(segs, times, t_us)
    cands = beat_fraction(time_to_beat(t_us, segs, times), beat_ms, allowed_dens=allowed_dens)
    ranked = [[w, n, d] for w, n, d, _ in cands]
    best_err = round(cands[0][3], 3) if cands else None
    return dict(t_us=t_us, beat=ranked[0] if ranked else None,
                alternatives=ranked[1:], err_ms=best_err,
                sure=bool(cands) and cands[0][3] < 0.6 and _family_class(cands[0][2]) < 3)


def convert_events(events, bpm_list, clock_shift_us=0, allowed_dens=None):
    """draft-events.json 的 events 列表 -> 编辑器友好结构(onset/end 都换算)。

    clock_shift_us:模型钟到编辑器钟的常数偏移(Phira: mp3 -2ms / ogg +10ms,按需传入;
    默认 0,即假定 BPM 图与模型在同一时钟)。
    """
    segs = parse_bpm_list(bpm_list)
    times = segment_times(segs)
    out = []
    for e in events:
        onset = convert_event(e['onset_us'] + clock_shift_us, segs, times, allowed_dens)
        same = e.get('end_us') is None or e.get('end_us') == e.get('onset_us')
        end = onset if same else convert_event(e['end_us'] + clock_shift_us, segs, times, allowed_dens)
        out.append(dict(type=e['type'], lane=e.get('lane'),
                        onset=onset['beat'], onset_us=e['onset_us'],
                        onset_err_ms=onset['err_ms'], onset_sure=onset['sure'],
                        onset_alternatives=onset['alternatives'],
                        end=end['beat'], end_us=e.get('end_us'),
                        end_err_ms=end['err_ms'], end_sure=end['sure'],
                        end_alternatives=end['alternatives']))
    return out


# ---------------- 自测 ----------------
def _selftest():
    segs = parse_bpm_list([dict(bpm=174.0, startTime=[0, 0, 1])])
    times = segment_times(segs)
    beat_ms = 60_000 / 174.0
    cases = [  # (位置[拍,分子,分母], 标签)
        ([0, 3, 4], '3/4 常规'), ([0, 1, 8], '1/8'), ([0, 1, 16], '1/16'),
        ([0, 1, 24], '1/24 三连'), ([0, 1, 20], '1/20 五连'), ([0, 1, 32], '1/32 密'),
        ([0, 7, 22], '7/22 怪'), ([0, 5, 21], '5/21 更怪'), ([3, 1, 2], '跨整拍 3+1/2'),
    ]
    ok = 0
    for (w, n, d), label in cases:
        t = round(beat_value(w, n, d) * beat_ms * 1000)  # 模型只给整数微秒
        got = convert_event(t, segs, times)
        exp = beat_value(w, n, d)
        act = beat_value(*got['beat']) if got['beat'] else None
        same = act is not None and abs(act - exp) * beat_ms < 0.6
        ok += same
        print(f'{label:<10} {t:>7}us -> {got["beat"]} (err {got["err_ms"]}ms) {"OK" if same else "MISMATCH"}')
    # 变速歌:两段 174 -> 240
    segs2 = parse_bpm_list([dict(bpm=174.0, startTime=[0, 0, 1]), dict(bpm=240.0, startTime=[8, 0, 1])])
    times2 = segment_times(segs2)
    t2 = times2[1] * 1e6 + round(60_000 / 240.0 / 16 * 1000)  # 第二段起点后 1/16 拍
    got2 = convert_event(t2, segs2, times2)
    print(f'变速段      {t2:>7.0f}us -> {got2["beat"]} (err {got2["err_ms"]}ms) {"OK" if got2["beat"] == [8, 1, 16] else "MISMATCH"}')
    ok += got2['beat'] == [8, 1, 16]
    # 带模型误差的音,两种工况:
    # 无限制:只承诺"真值在候选里"(数值巧合可能排前,如 1/15 抢 1/16),sure=False 交谱师确认;
    # 编辑器吸附集(常见格):数值巧合被排除,小误差时真值必排第一。
    snap = [1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48]  # 典型编辑器吸附菜单
    for shift in (1200, 3000):
        t_n = round(beat_value(0, 1, 16) * beat_ms * 1000 + shift)
        free = convert_event(t_n, segs, times)
        alts = [tuple(free['beat'])] + [tuple(a) for a in free['alternatives']]
        hit = (0, 1, 16) in alts
        snapped = convert_event(t_n, segs, times, allowed_dens=snap)
        snap_first = snapped['beat'] == [0, 1, 16]
        snap_note = '真值第一' if snap_first else '真值在候选' if (0, 1, 16) in [tuple(snapped['beat'])] + [tuple(a) for a in snapped['alternatives']] else '丢失!'
        passed = hit and not free['sure'] and snap_note != '丢失!' and (snap_first or shift > 2000)
        print(f'误差{shift // 1000}ms      -> 自由:首选{free["beat"]}含真值={hit} | 吸附集:{snapped["beat"]}({snap_note}) {"OK" if passed else "MISMATCH"}')
        ok += passed
    total = len(cases) + 3
    print(f'{ok}/{total} 通过')
    return ok == total


if __name__ == '__main__':
    raise SystemExit(0 if _selftest() else 1)
