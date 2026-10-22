"""P1 event model (candidate B): FIRST_CANDIDATE §2.2–2.7 without the selection field.

Rows (one sequence per core): R reference evidence | I 8 intent | U controls+mode |
G 2 s song summary | K known events | Y BOS + completed groups. Each of 8 layers runs
ONE softmax over [frozen multi-tap full-band audio K/V (GQA) ; row K/V]. Visibility:
R->R; I->R,I; U,G,K->R,I,U,G,K; Y->prefix + causal Y. Audio is read only by K rows
(t±2 s) and Y rows ([t-1 s, t+3 s)); time enters every timed row and audio key as
1D RoPE (seconds from the core start).

Heads: one pointer softmax over {END, 800 free 10 ms bins, COPY_TIME anchors}, each
bin scored from its own audio cell (query-weighted log-sum-exp over 8 bands) and
10 ms mel features; a 1 ms residual; a GRU per group initialised with audio read
back at the chosen time; type -> location (mania lane / Phira line, side, x as a
discretised logistic mixture) -> hold end (2 s block -> 10 ms -> 1 ms, block detail
revealed after the block step). Losses are teacher-forced, each field normalised
by its own count. Generation lives in sample.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from . import features as FE
from .batch import MAX_NOTES, END_BLOCK_CELLS

NEG = -1e9


@dataclass(frozen=True)
class P1Config:
    d: int = 512
    layers: int = 8
    heads: int = 8
    audio_kv_heads: int = 2
    ffn: int = 1536
    dropout: float = 0.1
    taps: tuple = (8, 16, 24)
    top_tap: int = 24
    tap_dim: int = 1024
    summary_dim: int = 1024
    ptr: int = 128
    gru: int = 256
    intent: int = 8
    mixture: int = 5
    k_window: tuple = (-2.0, 2.0)
    y_window: tuple = (-1.0, 3.0)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def rope(x, t, timed):
    """x [B,H,N,D]; t [B,N] seconds; untimed rows keep identity. Periods 20 ms .. 120 s."""
    d = x.shape[-1] // 2
    period = torch.logspace(math.log10(0.02), math.log10(120.0), d, device=x.device)
    ang = (t * timed)[:, None, :, None].float() * (2 * math.pi / period)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :d], x[..., d:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class Layer(nn.Module):
    def __init__(self, c: P1Config):
        super().__init__()
        self.c = c
        hd = c.d // c.heads
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        self.q, self.k, self.v, self.o = (nn.Linear(c.d, c.d, bias=False) for _ in range(4))
        self.ka = nn.Linear(c.tap_dim, c.audio_kv_heads * hd, bias=False)
        self.va = nn.Linear(c.tap_dim, c.audio_kv_heads * hd, bias=False)
        self.mix = nn.Parameter(torch.zeros(len(c.taps)))
        self.f1, self.f2 = nn.Linear(c.d, 2 * c.ffn, bias=False), nn.Linear(c.ffn, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, taps_ln, audio_t, row_t, row_timed, mask):
        c, (B, N, _) = self.c, x.shape
        hd = c.d // c.heads
        z = self.n1(x)
        q, k, v = (f(z).view(B, N, c.heads, hd).transpose(1, 2) for f in (self.q, self.k, self.v))
        q, k = rope(q, row_t, row_timed), rope(k, row_t, row_timed)
        audio = torch.einsum('t,btnd->bnd', self.mix.softmax(0).to(taps_ln.dtype), taps_ln)
        Na = audio.shape[1]
        ka, va = (f(audio).view(B, Na, c.audio_kv_heads, hd).transpose(1, 2) for f in (self.ka, self.va))
        ka = rope(ka, audio_t, torch.ones_like(audio_t))
        rep = c.heads // c.audio_kv_heads
        ka, va = ka.repeat_interleave(rep, 1), va.repeat_interleave(rep, 1)
        att = F.scaled_dot_product_attention(q, torch.cat([ka, k], 2), torch.cat([va, v], 2), attn_mask=mask)
        x = x + self.drop(self.o(att.transpose(1, 2).reshape(B, N, c.d)))
        a, b = self.f1(self.n2(x)).chunk(2, -1)
        return x + self.drop(self.f2(F.silu(a) * b))


def masked_ce(logits, target, valid, mask=None, name=''):
    """Mean CE over valid rows; logits [..., C]; invalid classes masked; returns (loss, n, correct)."""
    if mask is not None:
        logits = logits.masked_fill(~mask, NEG)
    if not valid.any():
        z = logits.sum() * 0
        return z, 0, 0
    lg, tg = logits[valid].float(), target[valid]
    if mask is not None:
        okm = mask[valid].gather(-1, tg[:, None])[:, 0]
        assert bool(okm.all()), f'{name}: {int((~okm).sum())} target(s) outside the legality mask, e.g. class {int(tg[~okm][0])}'
    loss = F.cross_entropy(lg, tg)
    return loss, int(tg.numel()), int((lg.argmax(-1) == tg).sum())


def logistic_mixture_nll(params, x, bins=256):
    """Discretised logistic mixture on [-1,1] (PixelCNN++ edges). params [N, 3M], x [N]."""
    M = params.shape[-1] // 3
    logit_pi, mu, log_s = params[:, :M], params[:, M:2 * M].tanh(), params[:, 2 * M:].clamp(-7, 3)
    xc = x[:, None]
    inv = torch.exp(-log_s)
    half = 1.0 / (bins - 1)
    cdf_hi = torch.sigmoid(inv * (xc + half - mu))
    cdf_lo = torch.sigmoid(inv * (xc - half - mu))
    p = torch.where(xc < -0.999, cdf_hi, torch.where(xc > 0.999, 1 - cdf_lo, cdf_hi - cdf_lo))
    lp = torch.log(p.clamp_min(1e-12)) + F.log_softmax(logit_pi, -1)
    return -torch.logsumexp(lp, -1)


class P1Model(nn.Module):
    def __init__(self, c: P1Config = P1Config()):
        super().__init__()
        self.c = c
        d, T = c.d, len(c.taps)
        self.tap_ln = nn.ModuleList(nn.LayerNorm(c.tap_dim) for _ in range(T))
        self.band = nn.Parameter(torch.randn(8, c.tap_dim) * 0.02)
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.layers))
        self.norm = RMSNorm(d)
        # row embeddings
        self.type_e = nn.Embedding(6, 64)  # 4 types, 4=unselected, 5=pad
        self.lane_e = nn.Embedding(5, 64)
        self.line_e = nn.Embedding(FE.MAX_LINES + 1, 64)
        self.side_e = nn.Embedding(3, 16)
        self.ev_proj = nn.Linear(64 * 3 + 16 + 4, d)  # + x, log dur, flag_a, flag_b
        self.ref_mix = nn.Parameter(torch.zeros(T))
        self.ref_band_q = nn.Parameter(torch.randn(c.tap_dim) * 0.02)
        self.ref_audio = nn.Linear(c.tap_dim, d)
        self.row_kind = nn.Embedding(6, d)
        self.intent = nn.Parameter(torch.randn(c.intent, d) * 0.02)
        self.ctrl_e = nn.Embedding(len(FE.CONTROL_FIELDS) * 32, d)
        self.mode_e = nn.Embedding(len(FE.MODES), d)
        self.allowed_p = nn.Linear(4, d)
        self.sum_ln = nn.LayerNorm(c.summary_dim)
        self.sum_band = nn.Linear(c.summary_dim, d // 8)
        self.sum_stats = nn.Linear(6, d)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)
        self.right_p = nn.Linear(8, d)
        # group / note embeddings (attribute-only feedback, design §4.2)
        self.note_p = nn.Linear(64 * 3 + 16 + 2, c.gru)
        self.group_p = nn.Linear(2 * c.gru, d)
        # pointer
        P = c.ptr
        self.ptr_mix = nn.Parameter(torch.zeros(T))
        self.ptr_proj = nn.Linear(c.tap_dim, P)
        self.fine = nn.Sequential(nn.Conv1d(128, P, 5, padding=2), nn.GELU(), nn.Conv1d(P, P, 3, padding=1))
        self.q_a, self.q_f, self.q_c = nn.Linear(d, P), nn.Linear(d, P), nn.Linear(d, P)
        self.end_head = nn.Linear(d, 1)
        self.rel_bias = nn.Embedding(64, 1)
        self.state_bias = nn.Embedding(3, 1)
        self.copy_key = nn.Sequential(nn.Linear(2 * P, P), nn.GELU(), nn.Linear(P, P))
        self.res_head = nn.Sequential(nn.Linear(P + 3 * P, 256), nn.GELU(), nn.Linear(256, 10))
        self.res_q = nn.Linear(d, P)
        # inner GRU
        H = c.gru
        self.gather_q = nn.Linear(d, P)
        self.right_g = nn.Linear(8, 16)
        self.init = nn.Linear(d + 2 * P + 16 + 4, H)
        self.start = nn.Parameter(torch.randn(H) * 0.02)
        self.gru_cell = nn.GRUCell(H, H)
        self.cont = nn.Linear(H, 2)
        self.type_h = nn.Linear(H, 4)
        self.type_c = nn.Embedding(4, H)
        self.lane_h = nn.Linear(H + 4, 4)
        self.line_q = nn.Linear(H, 64)
        self.line_tab = nn.Embedding(FE.MAX_LINES, 64)
        self.side_h = nn.Linear(H + 64, 2)
        self.x_h = nn.Linear(H + 64, 3 * c.mixture)
        self.loc_c = nn.Linear(64 + 16 + 1, H)
        self.blk_band = nn.Linear(c.summary_dim, 32)
        self.blk_q = nn.Linear(H, 256)
        self.blk_off = nn.Embedding(FE.MAX_END_BLOCKS, 1)
        self.det_proj = nn.Linear(c.tap_dim, P)
        self.det_q = nn.Linear(H + 256, P)
        self.det_sub = nn.Embedding(4, 1)
        self.end_fine = nn.Linear(H + P, 10)

    # ------------------------------------------------------------ embeddings
    def _events(self, ev, selected_flag=False):
        typ = ev[..., 1].long()
        typ = torch.where(typ < 0, torch.full_like(typ, 4), typ)
        lane = ev[..., 2].long().clamp(min=-1) + 1
        lane = torch.where(lane == 0, torch.full_like(lane, 4), lane - 1)
        line = (ev[..., 3].long() + 1).clamp(0, FE.MAX_LINES)
        side = (ev[..., 4].long() + 1).clamp(0, 2)
        scal = torch.stack([ev[..., 5], torch.log1p(ev[..., 6].clamp(min=0)), ev[..., 7], ev[..., 8]], -1)
        return self.ev_proj(torch.cat([self.type_e(typ), self.lane_e(lane), self.line_e(line), self.side_e(side), scal], -1))

    def _note_emb(self, typ, lane, line, side, x, dur):
        lane_i = torch.where(lane < 0, torch.full_like(lane, 4), lane)
        line_i = (line + 1).clamp(0, FE.MAX_LINES)
        side_i = (side + 1).clamp(0, 2)
        return self.note_p(torch.cat([self.type_e(typ.clamp(0, 3)), self.lane_e(lane_i), self.line_e(line_i),
                                      self.side_e(side_i), x[..., None], torch.log1p(dur.clamp(min=0))[..., None]], -1))

    @staticmethod
    def _right_feats(r):
        lane = r[..., 1].long()
        oh = F.one_hot(torch.where(lane < 0, torch.full_like(lane, 4), lane), 5).float()
        return torch.cat([torch.log1p(r[..., 0:1].clamp(min=0)), torch.log1p(r[..., 3:4].clamp(min=0)), oh,
                          (r[..., 2:3].clamp(min=-1) + 1) / FE.MAX_LINES], -1)

    def _rows(self, bt):
        c = self.c
        B, dev = bt['mode'].shape[0], bt['mode'].device
        # R: evidence + its own audio cell (query-pooled over bands, mixed over taps)
        ra = torch.stack([ln(bt['ref_audio'][:, :, j]) for j, ln in enumerate(self.tap_ln)], 2)
        ra = torch.einsum('t,brtnd->brnd', self.ref_mix.softmax(0).to(ra.dtype), ra) + self.band
        w = torch.einsum('brnd,d->brn', ra, self.ref_band_q).softmax(-1)
        R = self._events(bt['refs']) + self.ref_audio(torch.einsum('brn,brnd->brd', w, ra)) + self.row_kind.weight[0]
        I = (self.intent[None] + self.row_kind.weight[1]).expand(B, -1, -1)
        nf = len(FE.CONTROL_FIELDS)
        idx = torch.arange(nf, device=dev) * 32 + (bt['controls'] + 1).clamp(0, 31)
        U = torch.cat([self.ctrl_e(idx), (self.mode_e(bt['mode']) + self.allowed_p(bt['allowed_types'].float()))[:, None]], 1)
        U = U + self.row_kind.weight[2]
        s = self.sum_band(self.sum_ln(bt['summary'])).flatten(-2)
        G = s + self.sum_stats(bt['summary_stats']) + self.row_kind.weight[3]
        K = self._events(bt['known']) + self.row_kind.weight[4]
        rows = [R, I, U, G, K]
        lens = [x.shape[1] for x in rows]
        times = [bt['refs'][..., 0], torch.zeros(B, lens[1], device=dev), torch.zeros(B, lens[2], device=dev),
                 bt['summary_t'], bt['known'][..., 0]]
        timed = [bt['refs_valid'].float(), torch.zeros(B, lens[1], device=dev), torch.zeros(B, lens[2], device=dev),
                 bt['summary_valid'].float(), bt['known_valid'].float()]
        valid = [bt['refs_valid'], torch.ones(B, lens[1], dtype=torch.bool, device=dev),
                 torch.ones(B, lens[2], dtype=torch.bool, device=dev), bt['summary_valid'], bt['known_valid']]
        return rows, lens, times, timed, valid

    def _y_rows(self, bt, group_emb):
        B = bt['mode'].shape[0]
        Y = torch.cat([self.bos.expand(B, 1, -1), group_emb], 1) + self.row_kind.weight[5]
        Y = Y + self.right_p(self._right_feats(bt['right_rows']))
        t = torch.cat([torch.zeros(B, 1, device=Y.device), bt['grp_t']], 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=Y.device), bt['grp_valid']], 1)
        return Y, t, valid

    def group_embedding(self, bt):
        e = self._note_emb(bt['note_type'], bt['note_lane'], bt['note_line'], bt['note_side'], bt['note_x'], bt['note_dur'])
        nm = (torch.arange(MAX_NOTES, device=e.device)[None, None] < bt['grp_n'][..., None])[..., None]
        s = (e * nm).sum(2)
        m = e.masked_fill(~nm, NEG).max(2).values.masked_fill(~nm.any(2), 0)
        return self.group_p(torch.cat([s, m], -1))

    # ------------------------------------------------------------ masks
    def _mask(self, bt, lens, times, valid, n_y):
        c = self.c
        dev = bt['mode'].device
        B = bt['mode'].shape[0]
        kinds = torch.cat([torch.full((n,), k, device=dev) for k, n in enumerate(lens + [n_y])])
        q, k = kinds[:, None], kinds[None, :]
        allow = torch.zeros(len(kinds), len(kinds), dtype=torch.bool, device=dev)
        allow |= (q == 0) & (k == 0)
        allow |= (q == 1) & (k <= 1)
        allow |= (q >= 2) & (q <= 4) & (k <= 4)
        allow |= (q == 5) & (k <= 4)
        y0 = sum(lens)
        allow[y0:, y0:] = torch.ones(n_y, n_y, dtype=torch.bool, device=dev).tril()
        rv = torch.cat(valid, 1)
        rows = allow[None] & rv[:, None, :]
        eye = torch.eye(len(kinds), dtype=torch.bool, device=dev)[None]
        rows = torch.where(rv[:, :, None], rows, eye)
        # audio windows
        t_row = torch.cat(times, 1)
        cell_t = (torch.arange(FE.LOCAL_CELLS, device=dev) + 0.5) * FE.CELL_US / FE.SEC - FE.HALO_US / FE.SEC
        dt = cell_t[None, None, :] - t_row[:, :, None]
        kq = (kinds == 4)[None, :, None]
        yq = (kinds == 5)[None, :, None]
        win = (kq & (dt >= c.k_window[0]) & (dt <= c.k_window[1])) | (yq & (dt >= c.y_window[0]) & (dt < c.y_window[1]))
        win = win & bt['audio_valid'][:, None, :] & rv[:, :, None]
        win = win[..., None].expand(-1, -1, -1, 8).reshape(B, len(kinds), FE.LOCAL_CELLS * 8)
        return torch.cat([win, rows], -1)[:, None]

    # ------------------------------------------------------------ forward
    def encode(self, bt):
        """Transformer over all rows; returns Y hidden states [B, G+1, d] and pointer features."""
        c = self.c
        B = bt['mode'].shape[0]
        taps = torch.stack([ln(bt['audio'][:, j]) for j, ln in enumerate(self.tap_ln)], 1)  # [B,T,400,8,Dt]
        taps = (taps + self.band).flatten(2, 3)  # [B,T,3200,Dt]
        audio_t = ((torch.arange(FE.LOCAL_CELLS, device=taps.device) + 0.5) * FE.CELL_US / FE.SEC - FE.HALO_US / FE.SEC)
        audio_t = audio_t.repeat_interleave(8)[None].expand(B, -1)
        rows, lens, times, timed, valid = self._rows(bt)
        Y, ty, vy = self._y_rows(bt, self.group_embedding(bt))
        x = torch.cat(rows + [Y], 1)
        row_t = torch.cat(times + [ty], 1)
        row_timed = torch.cat(timed + [vy.float()], 1)
        mask = self._mask(bt, lens, times + [ty], valid + [vy], Y.shape[1])
        for layer in self.layers:
            x = layer(x, taps, audio_t, row_t, row_timed, mask)
        h = self.norm(x)[:, sum(lens):]
        ptr_audio = self.ptr_proj(torch.einsum('t,btnd->bnd', self.ptr_mix.softmax(0).to(taps.dtype), taps))
        ptr_audio = ptr_audio.view(B, FE.LOCAL_CELLS, 8, -1)
        fine = self.fine(bt['mel'].transpose(1, 2)).transpose(1, 2)  # [B,1600,P]
        return h, ptr_audio, fine, ty

    def time_logits(self, bt, h, ptr_audio, fine, t_prev):
        """Pointer over [END | 800 bins | A anchors] for each Y row; returns logits and legality mask."""
        B, Ny, _ = h.shape
        dev = h.device
        qa, qf, qc = self.q_a(h), self.q_f(h), self.q_c(h)
        P = qa.shape[-1]
        core_cells = ptr_audio[:, FE.HALO_US // FE.CELL_US:(FE.HALO_US + FE.CORE_US) // FE.CELL_US]  # [B,200,8,P]
        band = torch.einsum('byp,bcnp->bycn', qa, core_cells) / math.sqrt(P)
        cell_score = torch.logsumexp(band, -1).repeat_interleave(4, -1)  # [B,Ny,800]
        f0 = FE.HALO_US // FE.BIN_US
        core_fine = fine[:, f0:f0 + FE.CORE_BINS]
        fine_score = torch.einsum('byp,bkp->byk', qf, core_fine) / math.sqrt(P)
        bins = torch.arange(FE.CORE_BINS, device=dev)
        prev_ms = (t_prev * 1000).round()  # [B,Ny] ms from core start (BOS: -1)
        delta = (bins[None, None] * 10 - prev_ms[..., None]).clamp(min=0)
        bucket = (torch.log2(delta / 10 + 1) * 6).long().clamp(max=63)
        state = bt['cell_state'][:, f0 // 4:f0 // 4 + FE.CORE_BINS // 4].repeat_interleave(4, -1)
        bin_logits = cell_score + fine_score + self.rel_bias(bucket).squeeze(-1) + self.state_bias(state).squeeze(-1)[:, None]
        ms = bins[:, None] * 10 + torch.arange(10, device=dev)[None]  # [800,10]
        ok = bt['bin_ok'][:, None] & (ms[None, None] > prev_ms[..., None, None])  # [B,Ny,800,10]
        bin_mask = ok.any(-1)
        # COPY_TIME anchors
        a_ms = (bt['copy_t'] * 1000).round()
        a_bin = (a_ms / 10).long().clamp(0, FE.CORE_BINS - 1)
        a_cell = (FE.HALO_US // FE.CELL_US + a_bin // 4)
        a_audio = ptr_audio.mean(2).gather(1, a_cell[..., None].expand(-1, -1, P))
        a_fine = fine.gather(1, (f0 + a_bin)[..., None].expand(-1, -1, P))
        a_key = self.copy_key(torch.cat([a_audio, a_fine], -1))
        copy_logits = torch.einsum('byp,bap->bya', qc, a_key) / math.sqrt(P)
        copy_mask = bt['copy_valid'][:, None] & (a_ms[:, None] > prev_ms[..., None])
        logits = torch.cat([self.end_head(h), bin_logits, copy_logits], -1)
        mask = torch.cat([torch.ones(B, Ny, 1, dtype=torch.bool, device=dev), bin_mask, copy_mask], -1)
        return logits, mask, ok

    def residual_logits(self, h_sel, fine, b_sel):
        """h_sel [N,d], fine [N,1600,P] gathered per row, b_sel [N] chosen bin."""
        f0 = FE.HALO_US // FE.BIN_US
        idx = (f0 + b_sel[:, None] + torch.arange(-1, 2, device=b_sel.device)[None]).clamp(0, fine.shape[1] - 1)
        ctx = fine.gather(1, idx[..., None].expand(-1, -1, fine.shape[-1])).flatten(1)
        return self.res_head(torch.cat([self.res_q(h_sel), ctx], -1))

    def gru_init(self, h_sel, ptr_audio_sel, fine_sel, t_new, right_new, busy):
        """Audio read back at the chosen time (±3 cells × 8 bands) initialises the inner GRU."""
        P = ptr_audio_sel.shape[-1]
        cell = (FE.HALO_US // FE.CELL_US + (t_new * FE.SEC / FE.CELL_US).floor().long())
        idx = (cell[:, None] + torch.arange(-3, 4, device=cell.device)[None]).clamp(0, FE.LOCAL_CELLS - 1)
        win = ptr_audio_sel.gather(1, idx[..., None, None].expand(-1, -1, 8, P)).flatten(1, 2)  # [N,56,P]
        w = torch.einsum('np,nkp->nk', self.gather_q(h_sel), win).div(math.sqrt(P)).softmax(-1)
        pooled = torch.einsum('nk,nkp->np', w, win)
        frame = (FE.HALO_US // FE.BIN_US + (t_new * FE.SEC / FE.BIN_US).floor().long()).clamp(0, fine_sel.shape[1] - 1)
        f = fine_sel.gather(1, frame[:, None, None].expand(-1, 1, P))[:, 0]
        return torch.tanh(self.init(torch.cat([h_sel, pooled, f, self.right_g(self._right_feats(right_new)), busy.float()], -1)))

    def forward(self, bt):
        """Teacher-forced losses for a collated batch. Returns (total_loss, metrics)."""
        h, ptr_audio, fine, ty = self.encode(bt)
        B, Ny, d = h.shape
        G = Ny - 1
        dev = h.device
        t_prev = torch.cat([torch.full((B, 1), -0.001, device=dev), bt['grp_t']], 1)
        logits, mask, ok = self.time_logits(bt, h, ptr_audio, fine, t_prev)
        row_valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), bt['grp_valid']], 1)
        # target per Y row i: group i (bin / copy) or END after the last group
        tgt = torch.zeros(B, Ny, dtype=torch.long, device=dev)
        has = torch.cat([bt['grp_valid'], torch.zeros(B, 1, dtype=torch.bool, device=dev)], 1)
        copy = torch.cat([bt['grp_copy'], torch.full((B, 1), -1, device=dev)], 1)
        binx = torch.cat([bt['grp_bin'], torch.zeros(B, 1, dtype=torch.long, device=dev)], 1)
        tgt = torch.where(has & (copy >= 0), 1 + FE.CORE_BINS + copy, torch.where(has, 1 + binx, tgt))
        m = {}
        loss_t, n_t, c_t = masked_ce(logits, tgt, row_valid, mask, 'time')
        m['time'] = (loss_t, n_t, c_t)
        # 1 ms residual for free-bin targets
        free = bt['grp_valid'] & (bt['grp_copy'] < 0)
        bi, gi = free.nonzero(as_tuple=True)
        if len(bi):
            res = self.residual_logits(h[bi, gi], fine[bi], bt['grp_bin'][bi, gi])
            res_mask = ok[bi, gi, bt['grp_bin'][bi, gi]]
            m['residual'] = masked_ce(res, bt['grp_res'][bi, gi], torch.ones_like(bi, dtype=torch.bool), res_mask, 'residual')
        # inner GRU over every valid group (teacher forced)
        bi, gi = bt['grp_valid'].nonzero(as_tuple=True)
        if len(bi):
            self._inner(bt, h, ptr_audio, fine, bi, gi, m)
        total, metrics = 0.0, {}
        for k, (l, n, cnum) in m.items():
            if n:
                total = total + l
                metrics[k] = (float(l.detach()), n, cnum)
        return total, metrics

    def _inner(self, bt, h, ptr_audio, fine, bi, gi, m):
        N = len(bi)
        dev = h.device
        hs = self.gru_init(h[bi, gi], ptr_audio[bi], fine[bi], bt['grp_t'][bi, gi], bt['right_new'][bi, gi], bt['busy'][bi, gi])
        n_notes = bt['grp_n'][bi, gi]
        typ, lane, line = bt['note_type'][bi, gi], bt['note_lane'][bi, gi], bt['note_line'][bi, gi]
        side, x, dur = bt['note_side'][bi, gi], bt['note_x'][bi, gi], bt['note_dur'][bi, gi]
        mode = bt['mode'][bi]
        allowed = bt['allowed_types'][bi] & torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool, device=dev)[mode]
        n_lines = bt['lines'][bi]
        used = bt['busy'][bi, gi].clone()
        hold_pos = {(int(b), int(g), int(n)): k for k, (b, g, n) in enumerate(bt['hold_index'].tolist())}
        acc = {k: [] for k in ('cont', 'type', 'lane', 'line', 'side')}
        x_nll, hold_rows = [], []
        inp = self.start.expand(N, -1)
        for j in range(MAX_NOTES + 1):
            hs = self.gru_cell(inp, hs)
            active = j <= n_notes  # this step exists (decides continue or emits note j)
            if j >= 1:
                acc['cont'].append((self.cont(hs), (j < n_notes).long(), active, None))
            emit = j < n_notes
            if not emit.any():
                break
            jt = typ[:, j]
            acc['type'].append((self.type_h(hs), jt, emit, allowed))
            o1 = hs + self.type_c(jt)
            is_m, is_p = emit & (mode == 0), emit & (mode == 1)
            lane_mask = ~used
            acc['lane'].append((self.lane_h(torch.cat([o1, used.float()], -1)), lane[:, j].clamp(min=0), is_m, lane_mask))
            line_logits = self.line_q(o1) @ self.line_tab.weight.T
            line_mask = torch.arange(FE.MAX_LINES, device=dev)[None] < n_lines[:, None]
            acc['line'].append((line_logits, line[:, j].clamp(min=0), is_p, line_mask))
            le = self.line_tab(line[:, j].clamp(0, FE.MAX_LINES - 1))
            acc['side'].append((self.side_h(torch.cat([o1, le], -1)), side[:, j].clamp(min=0), is_p, None))
            if is_p.any():
                x_nll.append(logistic_mixture_nll(self.x_h(torch.cat([o1, le], -1))[is_p], x[is_p, j]))
            loc = torch.where(mode[:, None] == 0, torch.cat([F.one_hot(lane[:, j].clamp(min=0), 4).float(),
                                                             torch.zeros(N, 60, device=dev)], -1), le)
            o2 = o1 + self.loc_c(torch.cat([loc, self.side_e((side[:, j] + 1).clamp(0, 2)), x[:, j, None]], -1))
            for r in (emit & (jt == 1)).nonzero(as_tuple=True)[0].tolist():
                hold_rows.append((r, hold_pos[(int(bi[r]), int(gi[r]), j)], o2[r]))
            used = used | (is_m[:, None] & F.one_hot(lane[:, j].clamp(min=0), 4).bool())
            inp = self._note_emb(jt, lane[:, j], line[:, j], side[:, j], x[:, j], dur[:, j])
        for k, items in acc.items():
            if not items:
                continue
            lg = torch.cat([a for a, _, _, _ in items])
            tg = torch.cat([b for _, b, _, _ in items])
            vl = torch.cat([v for _, _, v, _ in items])
            mk = None if items[0][3] is None else torch.cat([mm for _, _, _, mm in items])
            m[k] = masked_ce(lg, tg, vl, mk, k)
        if x_nll:
            xn = torch.cat(x_nll)
            m['x'] = (xn.mean(), len(xn), 0)
        if hold_rows:
            self._hold(bt, hold_rows, m)

    def _hold(self, bt, hold_rows, m):
        dev = bt['mode'].device
        k = torch.tensor([r[1] for r in hold_rows], device=dev)
        o = torch.stack([r[2] for r in hold_rows])
        onset = bt['hold_onset_us'][k]
        lo, hi = bt['hold_ok_lo'][k], bt['hold_ok_hi'][k]
        b0 = onset // FE.BLOCK_US
        blocks = torch.arange(FE.MAX_END_BLOCKS, device=dev)
        bstart = (b0[:, None] + blocks[None]) * FE.BLOCK_US
        blk_mask = (bstart < hi[:, None] + 1) & (bstart + FE.BLOCK_US > lo[:, None])
        feats = self.blk_band(self.sum_ln(bt['hold_blocks'][k])).flatten(-2)  # [H,16,256]
        blk_logits = torch.einsum('hd,hkd->hk', self.blk_q(o), feats) / 16 + self.blk_off.weight.T
        ones = torch.ones(len(k), dtype=torch.bool, device=dev)
        m['end_block'] = masked_ce(blk_logits, bt['hold_block'][k], ones, blk_mask, 'end_block')
        # block detail is revealed only for the (teacher-forced) chosen block
        tb = bt['hold_block'][k]
        det = self.det_proj(self.tap_ln[self.c.taps.index(self.c.top_tap)](bt['hold_detail'][k]) + self.band)  # [H,50,8,P]
        q = self.det_q(torch.cat([o, feats.gather(1, tb[:, None, None].expand(-1, 1, feats.shape[-1]))[:, 0]], -1))
        cell = torch.logsumexp(torch.einsum('hp,hcnp->hcn', q, det) / math.sqrt(q.shape[-1]), -1)
        coarse_logits = cell.repeat_interleave(4, -1) + self.det_sub.weight.T.repeat(1, END_BLOCK_CELLS)
        start = (b0 + tb) * FE.BLOCK_US
        cb = start[:, None] + torch.arange(FE.BLOCK_US // FE.BIN_US, device=dev)[None] * FE.BIN_US
        coarse_mask = (cb <= hi[:, None]) & (cb + FE.BIN_US - FE.MS >= lo[:, None])  # some ms point in [lo, hi]
        m['end_coarse'] = masked_ce(coarse_logits, bt['hold_coarse'][k], ones, coarse_mask, 'end_coarse')
        cbin = start + bt['hold_coarse'][k] * FE.BIN_US
        fb = cbin[:, None] + torch.arange(10, device=dev)[None] * FE.MS
        fine_mask = (fb <= hi[:, None]) & (fb >= lo[:, None])  # exact ms end times; lo = onset + 1 us
        ctx = det.mean(2).gather(1, (bt['hold_coarse'][k] // 4)[:, None, None].expand(-1, 1, det.shape[-1]))[:, 0]
        m['end_fine'] = masked_ce(self.end_fine(torch.cat([o, ctx], -1)), bt['hold_fine'][k], ones, fine_mask, 'end_fine')
