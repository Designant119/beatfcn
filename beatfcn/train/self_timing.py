"""Self-timing (Track B1): estimate a song's beat grid from audio with the frozen encoder + grid head, then feed
it to the model as its timing input (the same beat features a charter's timing would give). Audio only: a
capability readout, not a charter input.

fit_grid: the head predicts per 10 ms frame the beat phase theta_t (and the bar phase). For a candidate beat
period P the coherence |sum_t w_t exp(i(theta_t - 2 pi t / P))| / sum_t w_t is 1 when the predicted phases
advance exactly one cycle per P; its angle gives the grid offset. A coarse BPM scan, a fine refinement and
local windows (tempo-change detection: one segment per stable stretch) give charter-style segments
[[start_us, beat_us, meter]]; the bar phase picks the downbeat. Low coherence -> no grid (cold fallback).
"""
import math

import numpy as np

BPM_RANGE = (60.0, 300.0)
MIN_COHERENCE = 0.5


def _coherence(t, theta, w, period):
    z = np.sum(w * np.exp(1j * (theta - 2 * np.pi * t / period)))
    return np.abs(z) / max(np.sum(w), 1e-9), np.angle(z)


def fit_constant(t, theta, w=None, *, bpm_range=BPM_RANGE, coarse_step=0.25, fine_step=0.005):
    """(beat_s, offset_s, coherence) of the best constant grid; offset = time of a beat (phase 0)."""
    t = np.asarray(t, np.float64); theta = np.asarray(theta, np.float64)
    w = np.ones_like(t) if w is None else np.asarray(w, np.float64)
    best = (0.0, None, 0.0)
    for bpm in np.arange(bpm_range[0], bpm_range[1] + 1e-9, coarse_step):
        c, ang = _coherence(t, theta, w, 60.0 / bpm)
        if c > best[0]:
            best = (c, bpm, ang)
    _, bpm0, _ = best
    if bpm0 is None:
        return None
    fine = (0.0, None, 0.0)
    for bpm in np.arange(bpm0 - coarse_step, bpm0 + coarse_step + 1e-9, fine_step):
        c, ang = _coherence(t, theta, w, 60.0 / bpm)
        if c > fine[0]:
            fine = (c, bpm, ang)
    c, bpm, ang = fine
    beat = 60.0 / bpm
    # theta_t ~ 2 pi (t - offset)/beat  =>  angle(z) = -2 pi offset / beat
    offset = (-ang / (2 * np.pi)) * beat % beat
    return beat, offset, c


def downbeat_offset(t, bar_theta, w, beat, offset, meter=4):
    """Which of the meter beats is the bar start, from the predicted bar phase."""
    best, pick = -np.inf, 0
    for k in range(meter):
        phase = 2 * np.pi * (t - (offset + k * beat)) / (beat * meter)
        score = np.sum(w * np.cos(bar_theta - phase))
        if score > best:
            best, pick = score, k
    return offset + pick * beat


def _segment(t, theta, bar, w, s0, beat, meter):
    r = fit_constant(t, theta, w, bpm_range=(60 / beat * 0.98, 60 / beat * 1.02), coarse_step=0.05, fine_step=0.002)
    if r is None:
        return None, 0.0
    o = downbeat_offset(t, bar, w, r[0], r[1], meter)
    bar_len = r[0] * meter
    first = o + math.floor((s0 - o) / bar_len) * bar_len
    return [first * 1e6, r[0] * 1e6, meter], r[2]


def fit_grid(times_s, pred, weights=None, *, window_s=24.0, hop_s=8.0, change_tol=0.004, min_coherence=MIN_COHERENCE,
             meter=4):
    """pred [T, >=4]: (beat_sin, beat_cos, bar_sin, bar_cos, ...). Returns (segments or None, diagnostics)."""
    t = np.asarray(times_s, np.float64)
    pred = np.asarray(pred, np.float64)
    theta = np.arctan2(pred[:, 0], pred[:, 1])
    bar = np.arctan2(pred[:, 2], pred[:, 3])
    w = np.hypot(pred[:, 0], pred[:, 1]) if weights is None else np.asarray(weights, np.float64)
    glob = fit_constant(t, theta, w)
    if glob is None:
        return None, dict(reason='no_fit')
    beat, offset, coh = glob
    local = []
    for s0 in np.arange(t.min(), max(t.min() + 1e-6, t.max() - window_s) + 1e-9, hop_s):
        m = (t >= s0) & (t < s0 + window_s)
        if m.sum() < 200:
            continue
        r = fit_constant(t[m], theta[m], w[m], coarse_step=0.5, fine_step=0.02)
        if r is not None:
            local.append((float(s0), r[0], r[2]))
    # octave-normalise local tempi to the global beat (a window locked on half / double tempo is the same tempo)
    local = [(s0, b * 2.0 ** round(math.log2(beat / b)), c) for s0, b, c in local]
    good = [x for x in local if x[2] >= min_coherence]
    diag = dict(bpm=60 / beat, coherence=float(coh), windows=len(local), coherent_windows=len(good))
    off = [abs(x[1] / beat - 1) > change_tol for x in good]
    # a tempo change = >= 2 CONSECUTIVE coherent windows away from the global tempo (a share threshold missed a
    # 250 -> 260 BPM change in the first ~22% of a song: DEV phira:56848)
    run = longest = 0
    for o in off:
        run = run + 1 if o else 0
        longest = max(longest, run)
    if coh >= min_coherence and longest < 2:
        seg, _ = _segment(t, theta, bar, w, float(t.min()), beat, meter)
        return ([seg] if seg else None), dict(diag, piecewise=1)
    if len(good) < max(2, 0.6 * max(1, len(local))):
        return None, dict(diag, reason='low_coherence')
    stretches = []
    for s0, b, c in good:
        if not stretches or abs(b / stretches[-1][1] - 1) > change_tol:
            stretches.append([s0, b])
    out = []
    for k, (s0, b) in enumerate(stretches):
        s1 = stretches[k + 1][0] if k + 1 < len(stretches) else t.max() + 1
        m = (t >= s0) & (t < s1)
        if m.sum() < 200:
            continue
        seg, c = _segment(t[m], theta[m], bar[m], w[m], s0, b, meter)
        if seg and c >= min_coherence:
            out.append(seg)
    return (out or None), dict(diag, piecewise=len(out))


def load_head(path, device):
    import torch
    from scripts.train_grid_head import make_head
    saved = torch.load(path, map_location='cpu', weights_only=False)
    head = make_head(int(saved['d']))
    head.load_state_dict(saved['state'])
    return head.to(device).eval()


def song_phases(model, head, features, audio_sha256, mode, duration_us, device, *, core_us=8_000_000, batch=16):
    """Cold encode every 8 s core of a song; (times_s [T], pred [T, 8]) for the valid 10 ms frames."""
    import torch
    from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
    from beatfcn.train.counterfactual_exposure import unconditional_inputs
    from beatfcn.train.local_panel import collate
    from beatfcn.train.local_request_panel import request_inputs
    tap = {}
    handle = model.onset_norm.register_forward_pre_hook(lambda _m, args: tap.__setitem__('x', args[0]))
    times, preds = [], []
    try:
        starts = list(range(0, int(duration_us), core_us))
        for k in range(0, len(starts), batch):
            rows, metas = [], []
            for s in starts[k:k + batch]:
                req = EventRequest(mode=mode, start_us=int(s), end_us=int(min(s + core_us, duration_us)),
                                   duration_us=int(duration_us), known_carry=())
                cond = unconditional_inputs(mode, req)
                inp, _ = request_inputs(features, audio_sha256, req, cond['controls'])
                inp.update(cond)
                rows.append(dict(inputs=inp, sequence=teacher_sequence(encode_events([], req), req)))
                metas.append((s, np.asarray(inp['mel_times'], np.float64), np.asarray(inp['mel_valid'], bool)))
            inputs, _ = collate(rows)
            inputs = {key: v.to(device) for key, v in inputs.items()}
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
                model.encode(inputs)
                out = head(tap['x'].float()).float().cpu().numpy()
            for b, (s, mt, mv) in enumerate(metas):
                n = len(mt)
                abs_t = s / 1e6 + mt
                keep = mv & (mt >= 0) & (mt < core_us / 1e6)   # only the core itself (context frames overlap)
                times.append(abs_t[keep]); preds.append(out[b, :n][keep])
    finally:
        handle.remove()
    return np.concatenate(times), np.concatenate(preds)


def self_timing_provider(model, head, features, songs, device):
    """local_song_rollout timing_provider from audio only: per song the fitted grid (None = cold fallback)."""
    grids, diags = {}, {}
    for song in songs:
        t, p = song_phases(model, head, features, song['audio_sha256'], song['mode'], song['duration_us'], device)
        segs, diag = fit_grid(t, p)
        grids[song['song_id']] = None if segs is None else np.asarray(segs, np.float64)
        diags[song['song_id']] = diag

    def provider(state, request):
        return grids.get(state['chart_id'])
    return provider, grids, diags


# ---- refine_grid: tighten a head-estimated grid with the model's own cold notes (no reference, no extra GPU) ----
# Human charts overwhelmingly use round tempi (92.7% integer BPM on 22.8k charts), and a 0.05% tempo error drifts
# past the scoring tolerance within a song. The cold notes tell which round tempo and phase the music locks to.
TOL_US = 8_000.0
SWITCH_PENALTY = 0.15
EST_MARGIN = 0.01


def _line_score(on, step_us, tol_us=TOL_US, bin_us=1_000.0):
    """(share of onsets within tol of a line spaced step_us, offset_us of the best such line set)."""
    if not len(on):
        return 0.0, 0.0
    n = max(1, int(round(step_us / bin_us)))
    hist = np.bincount(((on % step_us) / step_us * n).astype(np.int64) % n, minlength=n).astype(np.float64)
    r = max(1, int(round(tol_us / step_us * n)))
    kernel = np.zeros(n)
    kernel[np.arange(-r, r + 1) % n] = 1
    conv = np.real(np.fft.ifft(np.fft.fft(hist) * np.fft.fft(kernel)))
    k = int(np.argmax(conv))
    return float(conv[k] / len(on)), (k + .5) * step_us / n


def _candidates(segs):
    main = max(segs, key=lambda s: s[1])[1] if len(segs) == 1 else _main_beat(segs)
    out = {60e6 / main}
    for s in segs:
        b = s[1] * 2.0 ** round(math.log2(main / s[1]))
        out.add(60e6 / b)
        out.add(float(round(60e6 / b)))
        out.add(round(120e6 / b) / 2.0)
    bpm = 60e6 / main
    out.update(float(x) for x in range(int(math.ceil(bpm * .92)), int(bpm * 1.08) + 1))
    return sorted(out)


def _main_beat(segs):
    if len(segs) == 1:
        return segs[0][1]
    span = [segs[i + 1][0] - segs[i][0] for i in range(len(segs) - 1)] + [0.0]
    return segs[int(np.argmax(span))][1]


def _beat_rotation(on, beat, offset, prefer=None):
    """Which of the 4 line sets is the beat: the est phase when given, else the one with most notes on it."""
    step = beat / 4
    opts = [(offset + k * step) % beat for k in range(4)]
    if prefer is not None:
        d = [abs(((o - prefer + beat / 2) % beat) - beat / 2) for o in opts]
        return opts[int(np.argmin(d))]
    return opts[int(np.argmax([_line_score_at(on, beat, o) for o in opts]))]


def _line_score_at(on, step, offset, tol=TOL_US):
    return float(np.mean(np.abs((on - offset + step / 2) % step - step / 2) <= tol)) if len(on) else 0.0


def grid_score(segs, on, duration_us, tol=TOL_US):
    """Share of onsets within tol of a 1/4-beat line of the piecewise grid."""
    on = np.asarray(on, np.float64)
    if not len(on) or not segs:
        return 0.0
    hit = 0
    for i, (st, beat, _m) in enumerate(segs):
        lo = st if i else -math.inf
        hi = segs[i + 1][0] if i + 1 < len(segs) else math.inf
        x = on[(on >= lo) & (on < hi)]
        hit += _line_score_at(x, beat / 4, st, tol) * len(x)
    return hit / len(on)


def refine_grid(segs, onsets_us, duration_us, *, window_us=16e6, hop_us=8e6, meter=None):
    """Return (segments, diagnostics). segments keeps the [start_us, beat_us, meter] form; never scores worse than est."""
    on = np.sort(np.asarray(onsets_us, np.float64))
    if not segs or len(on) < 16:
        return segs, dict(changed=False, reason='no_input')
    meter = meter or int(segs[0][2])
    est_score = grid_score(segs, on, duration_us)
    cands = _candidates(segs)
    starts = np.arange(0.0, max(hop_us, duration_us - window_us + hop_us), hop_us)
    table = np.zeros((len(starts), len(cands)))
    for w, s0 in enumerate(starts):
        x = on[(on >= s0) & (on < s0 + window_us)]
        for c, bpm in enumerate(cands):
            table[w, c] = _line_score(x, 60e6 / bpm / 4)[0]
    # prefer round tempi on ties
    round_bonus = np.array([0.002 if abs(b - round(b)) < 1e-6 else 0.0 for b in cands])
    table = table + round_bonus
    best = table[0].copy()
    back = np.zeros_like(table, dtype=np.int64)
    for w in range(1, len(starts)):
        stay = best
        jump = best.max() - SWITCH_PENALTY
        k = int(np.argmax(best))
        back[w] = np.where(stay >= jump, np.arange(len(cands)), k)
        best = np.maximum(stay, jump) + table[w]
    path = [int(np.argmax(best))]
    for w in range(len(starts) - 1, 0, -1):
        path.append(int(back[w, path[-1]]))
    path = path[::-1]
    stretches = []
    for w, c in enumerate(path):
        if not stretches or stretches[-1][1] != c:
            stretches.append([float(starts[w]), c])
    edges = [0.0]
    for k in range(1, len(stretches)):  # a change point: the split inside the overlap that keeps the most notes on lines
        a, b = 60e6 / cands[stretches[k - 1][1]], 60e6 / cands[stretches[k][1]]
        lo, hi = stretches[k][0] - hop_us, stretches[k][0] + window_us
        best_t, best_v = stretches[k][0], -1.0
        for t in np.arange(max(edges[-1], lo), hi, 250e3):
            xa, xb = on[(on >= lo) & (on < t)], on[(on >= t) & (on < hi)]
            v = _line_score(xa, a / 4)[0] * len(xa) + _line_score(xb, b / 4)[0] * len(xb)
            if v > best_v:
                best_t, best_v = float(t), v
        edges.append(best_t)
    out = []
    est_main = _main_beat(segs)
    for k, (_s, c) in enumerate(stretches):
        lo, hi = edges[k], edges[k + 1] if k + 1 < len(edges) else math.inf
        x = on[(on >= lo) & (on < hi)]
        beat = 60e6 / cands[c]
        _, off = _line_score(x, beat / 4)
        prefer = None
        if abs(beat / est_main - 1) < 0.02:  # same tempo as the head's: keep its beat, only fix the fine phase
            mid = np.median(x) if len(x) else lo
            prefer = (segs[0][0] + round((mid - segs[0][0]) / est_main) * est_main) % beat
            prefer = ((mid - (mid - prefer) % beat)) % beat
        off = _beat_rotation(x, beat, off, prefer)
        first = off + math.floor((lo - off) / (beat * meter)) * (beat * meter) if k == 0 else lo + ((off - lo) % beat)
        out.append([float(first), float(beat), meter])
    new_score = grid_score(out, on, duration_us)
    diag = dict(changed=True, est_score=round(est_score, 4), score=round(new_score, 4), segments=len(out),
                bpm=[round(60e6 / s[1], 3) for s in out])
    if new_score < est_score + EST_MARGIN:
        return segs, dict(diag, changed=False, reason='no_gain')
    return out, diag
