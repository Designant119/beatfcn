"""Collate CoreExamples + song caches into padded torch tensors (P1).

Song caches (any array-like, typically np.memmap): taps {layer: [cells, 8, Dt]},
summary [S, 8, 1024] (2 s means of H20), mel10 [frames, 128]. Only slices a core
needs are read: 16 s of taps and mel, the reference cells, and for each target
hold its 16 candidate end blocks (summary) plus the true end block's detail
(top tap only, revealed AFTER the block prediction, design §4.3).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from . import features as F

MAX_NOTES = 8
END_BLOCK_CELLS = F.BLOCK_US // F.CELL_US  # 50
FRAMES_LOCAL = F.LOCAL_CELLS * F.FRAMES_PER_CELL  # 1600


@dataclass
class SongCache:
    taps: dict  # layer -> [cells, 8, Dt]
    summary: np.ndarray  # [S, 8, 1024]
    mel10: np.ndarray  # [frames, 128]
    duration_us: int


def _slice(arr, start, n, fill=0.0, dtype=np.float32):
    """Rows [start, start+n) of arr with zero (or fill) outside; returns ndarray and validity.
    Int8 PCA taps (anything with .raw) are sliced raw: dequantisation happens on the device (dequant)."""
    src = getattr(arr, 'raw', arr)
    if src is not arr:
        dtype = np.int8
    out = np.full((n, *arr.shape[1:]), fill, dtype)
    a, b = max(start, 0), min(start + n, arr.shape[0])
    if b > a:
        out[a - start:b - start] = np.asarray(src[a:b], dtype)
    valid = np.zeros(n, bool)
    valid[max(a - start, 0):max(b - start, 0)] = True
    return out, valid


def pool_summary(summary: np.ndarray, max_tokens: int = F.MAX_SUMMARY):
    """Adaptive bucket width for long songs (design §5.1); returns tokens and their centre time (s)."""
    s = np.asarray(summary, np.float32)
    k = -(-len(s) // max_tokens)
    n = -(-len(s) // k)
    pad = np.zeros((n * k, *s.shape[1:]), np.float32)
    pad[:len(s)] = s
    cnt = np.minimum(k, len(s) - np.arange(n) * k)
    tokens = pad.reshape(n, k, *s.shape[1:]).sum(1) / cnt[:, None, None]
    centre = (np.arange(n) * k + cnt / 2) * F.BLOCK_US / F.SEC
    return tokens, centre, k


BUCKET = {'G': 32, 'K': 64, 'R': 64, 'A': 16, 'S': 64, 'H': 64}  # variable dims rounded up: few distinct batch shapes


def _bucket(n, q):
    return -(-max(n, 1) // q) * q


def collate(examples: list[F.CoreExample], songs: list[SongCache], taps: tuple[int, ...], top_tap: int) -> dict:
    B = len(examples)
    G = _bucket(max([len(e.groups) for e in examples] + [1]), BUCKET['G'])
    K = _bucket(max([len(e.known) for e in examples] + [1]), BUCKET['K'])
    R = _bucket(max([len(e.refs) for e in examples] + [1]), BUCKET['R'])
    A = _bucket(max([len(e.copy_times) for e in examples] + [1]), BUCKET['A'])
    pooled = [pool_summary(s.summary) for s in songs]
    S = _bucket(max(len(p[0]) for p in pooled), BUCKET['S'])
    Dt = songs[0].taps[taps[0]].shape[-1]
    quant = all(hasattr(s.taps[l], 'raw') for s in songs for l in taps)
    if not quant and any(hasattr(s.taps[l], 'raw') for s in songs for l in taps):
        raise ValueError('mixed int8 / float tap caches in one batch')
    tdt = torch.int8 if quant else torch.float32
    out = {
        'mode': torch.tensor([e.mode for e in examples]),
        'audio': torch.zeros(B, len(taps), F.LOCAL_CELLS, 8, Dt, dtype=tdt), 'audio_valid': torch.zeros(B, F.LOCAL_CELLS, dtype=torch.bool),
        'mel': torch.zeros(B, FRAMES_LOCAL, 128, dtype=torch.float16), 'cell_state': torch.zeros(B, F.LOCAL_CELLS, dtype=torch.long),
        'summary': torch.zeros(B, S, 8, songs[0].summary.shape[-1], dtype=torch.float16), 'summary_t': torch.zeros(B, S),
        'summary_stats': torch.zeros(B, S, 6), 'summary_valid': torch.zeros(B, S, dtype=torch.bool),
        'known': torch.zeros(B, K, 9), 'known_valid': torch.zeros(B, K, dtype=torch.bool),
        'refs': torch.zeros(B, R, 9), 'refs_valid': torch.zeros(B, R, dtype=torch.bool),
        'ref_audio': torch.zeros(B, R, len(taps), 8, Dt, dtype=tdt),
        'controls': torch.full((B, len(F.CONTROL_FIELDS)), -1, dtype=torch.long),
        'allowed_types': torch.zeros(B, 4, dtype=torch.bool), 'lines': torch.zeros(B, dtype=torch.long),
        'bin_ok': torch.zeros(B, F.CORE_BINS, 10, dtype=torch.bool),
        'copy_t': torch.zeros(B, A), 'copy_valid': torch.zeros(B, A, dtype=torch.bool),
        'right_rows': torch.zeros(B, G + 1, 4),
        'grp_valid': torch.zeros(B, G, dtype=torch.bool), 'grp_t': torch.zeros(B, G), 'grp_bin': torch.zeros(B, G, dtype=torch.long),
        'grp_res': torch.zeros(B, G, dtype=torch.long), 'grp_copy': torch.full((B, G), -1, dtype=torch.long),
        'grp_n': torch.zeros(B, G, dtype=torch.long), 'busy': torch.zeros(B, G, 4, dtype=torch.bool), 'right_new': torch.zeros(B, G, 4),
        'note_type': torch.zeros(B, G, MAX_NOTES, dtype=torch.long), 'note_lane': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long),
        'note_line': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long), 'note_side': torch.full((B, G, MAX_NOTES), -1, dtype=torch.long),
        'note_x': torch.zeros(B, G, MAX_NOTES), 'note_dur': torch.zeros(B, G, MAX_NOTES),
        'cell_sel': torch.zeros(B, F.CORE_US // F.CELL_US, dtype=torch.long),
    }
    out['max_notes'] = max([len(tg['notes']) for e in examples for tg in e.groups] + [0])  # host int: GRU loop length
    hold_index, hold = [], {'block': [], 'coarse': [], 'fine': [], 'ok_lo': [], 'ok_hi': [], 'onset_us': [], 'blocks': [], 'detail': []}
    for b, (ex, song) in enumerate(zip(examples, songs)):
        for j, layer in enumerate(taps):
            a, v = _slice(song.taps[layer], ex.song_cell0, F.LOCAL_CELLS)
            out['audio'][b, j] = torch.from_numpy(a)
        out['audio_valid'][b] = torch.from_numpy(v)
        m, _ = _slice(song.mel10, ex.song_cell0 * F.FRAMES_PER_CELL, FRAMES_LOCAL, dtype=np.float16)
        out['mel'][b] = torch.from_numpy(m)
        out['cell_state'][b] = torch.from_numpy(ex.cell_state)
        tok, centre, _ = pooled[b]
        out['summary'][b, :len(tok)] = torch.from_numpy(tok.astype(np.float16))
        out['summary_t'][b, :len(tok)] = torch.from_numpy(centre - ex.c0_us / F.SEC)
        stats = ex.summary_stats
        k = -(-len(stats) // F.MAX_SUMMARY)
        if k > 1:  # pool stats like the audio summary
            stats = np.stack([stats[i:i + k].mean(0) for i in range(0, len(stats), k)])
        n = min(len(stats), len(tok))
        out['summary_stats'][b, :n] = torch.from_numpy(stats[:n])
        out['summary_valid'][b, :len(tok)] = True
        out['known'][b, :len(ex.known)] = torch.from_numpy(ex.known)
        out['known_valid'][b, :len(ex.known)] = True
        out['refs'][b, :len(ex.refs)] = torch.from_numpy(ex.refs)
        out['refs_valid'][b, :len(ex.refs)] = True
        if len(ex.refs):
            cells = np.rint((ex.c0_us + ex.refs[:, 0].astype(np.float64) * F.SEC) / F.CELL_US).astype(np.int64)
            order = np.argsort(cells, kind='stable')
            breaks = np.flatnonzero(np.diff(cells[order]) > 200) + 1  # one contiguous read per reference cluster
            for grp in np.split(order, breaks):
                lo, hi = int(cells[grp].min()), int(cells[grp].max()) + 1
                for j, layer in enumerate(taps):
                    span, _ = _slice(song.taps[layer], lo, hi - lo)
                    out['ref_audio'][b, torch.from_numpy(grp), j] = torch.from_numpy(span[cells[grp] - lo])
        out['controls'][b] = torch.from_numpy(ex.controls)
        out['allowed_types'][b] = torch.from_numpy(ex.allowed_types)
        out['lines'][b] = ex.lines
        out['bin_ok'][b] = torch.from_numpy(ex.bin_ok)
        out['copy_t'][b, :len(ex.copy_times)] = torch.from_numpy(ex.copy_times / F.SEC).float()
        out['copy_valid'][b, :len(ex.copy_times)] = True
        out['right_rows'][b, :len(ex.right)] = torch.from_numpy(ex.right)
        # selection labels per core cell: any head (target or observed) -> 1; cells outside the audio -> -1
        core_cells = F.CORE_US // F.CELL_US
        lab = torch.zeros(core_cells, dtype=torch.long)
        for tg in ex.groups:
            lab[tg['t_rel_us'] // F.CELL_US] = 1
        for row in ex.known:
            t = row[0] * F.SEC
            if 0 <= t < F.CORE_US and row[7] == 0:
                lab[int(t // F.CELL_US)] = 1
        h0 = F.HALO_US // F.CELL_US
        lab[~out['audio_valid'][b, h0:h0 + core_cells]] = -1
        out['cell_sel'][b] = lab
        for g, tg in enumerate(ex.groups):
            if len(tg['notes']) > MAX_NOTES:
                raise F.SkipExample('group has more notes than MAX_NOTES')
            out['grp_valid'][b, g] = True
            out['grp_t'][b, g] = tg['t_rel_us'] / F.SEC
            out['grp_bin'][b, g], out['grp_res'][b, g], out['grp_copy'][b, g] = tg['bin'], tg['residual'], tg['copy']
            out['grp_n'][b, g] = len(tg['notes'])
            out['busy'][b, g] = torch.from_numpy(tg['busy_lanes'])
            out['right_new'][b, g] = torch.from_numpy(tg['right_new'])
            onset_abs = ex.c0_us + tg['t_rel_us']
            for n, nt in enumerate(tg['notes']):
                out['note_type'][b, g, n] = nt['type']
                out['note_lane'][b, g, n], out['note_line'][b, g, n] = nt['lane'], nt['line']
                out['note_side'][b, g, n], out['note_x'][b, g, n] = nt['side'], nt['x']
                if 'end_rel_us' not in nt:
                    continue
                end_abs = onset_abs + nt['end_rel_us']
                b0 = onset_abs // F.BLOCK_US
                blk = end_abs // F.BLOCK_US - b0
                if blk >= F.MAX_END_BLOCKS:
                    raise F.SkipExample('hold longer than the endpoint block range')
                out['note_dur'][b, g, n] = nt['end_rel_us'] / F.SEC
                hold_index.append((b, g, n))
                hold['block'].append(blk)
                hold['coarse'].append(end_abs % F.BLOCK_US // F.BIN_US)
                hold['fine'].append(end_abs % F.BIN_US // F.MS)
                lo, hi = nt['end_ok_us']
                hold['ok_lo'].append(onset_abs + lo)
                hold['ok_hi'].append(onset_abs + hi)
                hold['onset_us'].append(onset_abs)
                bl, _ = _slice(song.summary, b0, F.MAX_END_BLOCKS, dtype=np.float16)
                hold['blocks'].append(bl)
                det, _ = _slice(song.taps[top_tap], (b0 + blk) * END_BLOCK_CELLS, END_BLOCK_CELLS)
                hold['detail'].append(det)
    H, Hp = len(hold_index), _bucket(len(hold_index), BUCKET['H'])
    pad = Hp - H  # padded holds: index (0,0,0), zero targets/blocks/detail, hold_valid False (masked everywhere)
    out['hold_index'] = torch.tensor(hold_index + [(0, 0, 0)] * pad, dtype=torch.long).reshape(-1, 3)
    out['hold_valid'] = torch.arange(Hp) < H
    for key in ('block', 'coarse', 'fine', 'ok_lo', 'ok_hi', 'onset_us'):
        out['hold_' + key] = torch.tensor(hold[key] + [0] * pad, dtype=torch.long)
    blocks = torch.zeros(Hp, F.MAX_END_BLOCKS, 8, songs[0].summary.shape[-1], dtype=torch.float16)
    detail = torch.zeros(Hp, END_BLOCK_CELLS, 8, Dt, dtype=tdt)
    if H:
        blocks[:H] = torch.from_numpy(np.stack(hold['blocks']))
        detail[:H] = torch.from_numpy(np.stack(hold['detail']))
    out['hold_blocks'], out['hold_detail'] = blocks, detail
    if quant:  # per-tap dequantisation scale [T, Dt]; applied on the device by dequant()
        sc = [np.broadcast_to(np.asarray(songs[0].taps[l].scale, np.float32), (Dt,)) for l in taps]
        out['tap_scale'] = torch.from_numpy(np.stack(sc).copy())
        out['tap_top'] = taps.index(top_tap)
    return out


def dequant(bt: dict) -> dict:
    """Device-side expansion of the compact transport format: int8 taps x scale, float16 -> float32.
    Idempotent; collate keeps batches small so worker IPC, pinning and H2D copies stay cheap."""
    sc = bt.pop('tap_scale', None)
    if sc is not None:
        top = bt.pop('tap_top', None)
        bt['audio'] = bt['audio'].float() * sc[None, :, None, None, :]
        bt['ref_audio'] = bt['ref_audio'].float() * sc[None, None, :, None, :]
        if top is not None:
            bt['hold_detail'] = bt['hold_detail'].float() * sc[top]
    for k, v in bt.items():
        if torch.is_tensor(v) and v.dtype == torch.float16:
            bt[k] = v.float()
    return bt
