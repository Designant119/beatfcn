"""Whole-window masked tuple model; no count inventory or recurrent members."""
import math

import numpy as np
import torch
from torch import nn

from .local_bundle import read_grid


class EventLayer(nn.Module):
    def __init__(self, d=256, heads=8, ffn=768):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, context, valid):
        q = self.norms[0](x)
        x = x + self.self_attention(q, q, q, need_weights=False)[0]
        x = x + self.cross_attention(self.norms[1](x), context, context,
                                     key_padding_mask=~valid, need_weights=False)[0]
        return x + self.ffn(self.norms[2](x))


class MaskedNativeEvents(nn.Module):
    def __init__(self, config, *, density_reference=None):
        super().__init__()
        self.c = config
        d = config.d
        self.register_buffer('periods', torch.logspace(math.log10(.02), math.log10(512.), 16))
        self.local_projection = nn.Linear(config.taps*config.bands*config.tap_dim, d)
        self.mel_projection = nn.Linear(config.mel_bins, d)
        self.song_projection = nn.Linear(config.tap_dim, d)
        self.local_convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k//2) for k in (5, 9)])
        self.mel_convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k//2) for k in (5, 9)])
        self.audio_time = nn.Linear(35, d)
        self.modality = nn.Embedding(5, d)
        self.reference_projection = nn.Linear(config.reference_dim, d)
        self.control_projection = nn.Linear(config.control_dim, d)
        self.control_ids = nn.Embedding(config.controls, d)
        self.mode = nn.Embedding(2, d)
        self.position = nn.Embedding(256, d)
        self.status = nn.Embedding(3, d)  # MASK,NULL,active
        self.kind_embedding = nn.Embedding(4, 16)
        self.event_projection = nn.Linear(88, d)
        self.layers = nn.ModuleList([EventLayer(d, config.heads, config.ffn) for _ in range(4)])
        self.norm = nn.LayerNorm(d)
        self.active_read = nn.Linear(d, 1)
        self.head_feature = nn.Sequential(nn.Linear(2*d+165, d), nn.GELU(), nn.LayerNorm(d))
        self.head_query = nn.Linear(d, d)
        self.head_key = nn.Linear(d, d)
        self.type_read = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 4))
        self.tail_query = nn.Linear(2*d, d)
        self.tail_key = nn.Linear(d+165, d)
        self.tail_level = nn.Embedding(3, d)
        self.tail_residual = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.tail_residual.weight)
        self.query_coverage = None

    def time_features(self, seconds):
        angles = seconds[..., None] * (2*math.pi/self.periods)
        return torch.cat((seconds[..., None]/1200, angles.sin(), angles.cos()), -1)

    def atom_features(self, us, origin, device):
        value = np.asarray(us, dtype=np.int64)
        exact = np.stack((((value-origin) % 40000)/40000, (value % 1000)/1000,
                          (value % 1000 != 0).astype(float)), -1)
        return torch.tensor(exact, dtype=torch.float32, device=device)

    def _conv(self, x, valid, layers):
        x = x.masked_fill(~valid[..., None], 0)
        for layer in layers:
            x = (x + torch.nn.functional.gelu(layer(x.transpose(1, 2)).transpose(1, 2))).masked_fill(~valid[..., None], 0)
        return x

    def _clock(self, times, span):
        return self.audio_time(torch.cat((self.time_features(times), span/1200), -1))

    def encode(self, inputs, layout, origin):
        device = inputs['mode'].device
        local = inputs['local'].permute(0, 2, 1, 3, 4).flatten(-3)
        local = self._conv(self.local_projection(local), inputs['local_valid'], self.local_convs)
        mel = self._conv(self.mel_projection(inputs['mel']), inputs['mel_valid'], self.mel_convs)
        valid_indices = torch.nonzero(inputs['song_valid'][0]).flatten()
        if not len(valid_indices):
            raise ValueError('No whole-song audio context')
        songs, clocks, spans = [], [], []
        for start in range(0, len(valid_indices), 4):
            ids = valid_indices[start:start+4]
            songs.append(inputs['song'][:, ids].mean(1))
            clock = inputs['song_times'][:, ids]
            clocks.append(clock.mean(1))
            spans.append(torch.stack((clock[:, 0], clock[:, -1]), -1))
        song = self.song_projection(torch.stack(songs, 1))
        song_times = torch.stack(clocks, 1)
        song_spans = torch.stack(spans, 1)
        song_valid = torch.ones_like(song_times, dtype=torch.bool)
        contexts = []
        masks = []
        for k, (values, times, valid, half) in enumerate([
            (local, inputs['local_times'], inputs['local_valid'], .02),
            (mel, inputs['mel_times'], inputs['mel_valid'], .005),
            (song, song_times, song_valid, None),
        ]):
            span = song_spans if half is None else torch.stack((times-half, times+half), -1)
            contexts.append(values + self._clock(times, span) + self.modality.weight[k])
            masks.append(valid)
        ref = self.reference_projection(inputs['reference'])
        ref = ref+self._clock(inputs['reference_times'], torch.stack((inputs['reference_times'], inputs['reference_times']), -1))+self.modality.weight[3]
        contexts.append(ref)
        masks.append(inputs['reference_valid'])
        controls = self.control_projection(inputs['controls']) + self.control_ids.weight[None] + self.modality.weight[4]
        contexts.extend((controls, self.mode(inputs['mode'])[:, None]))
        masks.extend((torch.ones(1, controls.shape[1], device=device, dtype=torch.bool), torch.ones(1, 1, device=device, dtype=torch.bool)))
        memory = dict(context=torch.cat(contexts, 1), valid=torch.cat(masks, 1), local=local,
                      song=song, song_times=song_times, song_spans=song_spans, song_valid=song_valid,
                      inputs=inputs, origin=int(origin), layout=layout)
        values = np.asarray(layout['heads'], np.int64)
        if len(values):
            memory['head_features'] = self.address_features(memory, values, 'legal_head_candidates')
            memory['head_keys'] = self.head_key(memory['head_features'])
        else:
            memory['head_features'] = local.new_empty((0, self.c.d))
            memory['head_keys'] = local.new_empty((0, self.c.d))
        return memory

    def _coverage(self, memory, values, role):
        if self.query_coverage is None or not len(values):
            return
        times = (np.asarray(values).reshape(-1)-memory['origin'])/1e6
        sources = [('mel', memory['inputs']['mel_times'], memory['inputs']['mel_valid'], .005),
                   ('song', memory['song_times'], memory['song_valid'], 1.)]
        if 'coverage_bounds' not in memory:
            memory['coverage_bounds']={}
            for name,clock,valid,padding in sources:
                good=clock[valid].detach().cpu().numpy()
                memory['coverage_bounds'][name]=(float(good.min())-padding,float(good.max())+padding)
        for name, clock, valid, padding in sources:
            low,high=memory['coverage_bounds'][name]
            key = role+'_'+name
            row = self.query_coverage.setdefault(key, dict(queries=0, in_range=0, min_s=float(times.min()), max_s=float(times.max())))
            row['queries'] += len(times)
            row['in_range'] += int(((times>=low)&(times<=high)).sum())
            row['min_s'] = min(row['min_s'], float(times.min()))
            row['max_s'] = max(row['max_s'], float(times.max()))

    def address_features(self, memory, us, role):
        inputs = memory['inputs'];device = inputs['mode'].device
        values = np.asarray(us, np.int64)
        times = torch.tensor((values-memory['origin'])/1e6, device=device, dtype=torch.float32)[None]
        local = read_grid(memory['local'], inputs['local_times'], inputs['local_valid'], times, .04)[0]
        raw = read_grid(inputs['mel'], inputs['mel_times'], inputs['mel_valid'], times, .01)[0]
        song = read_grid(memory['song'], memory['song_times'], memory['song_valid'], times, 2.)[0]
        mel_lo = inputs['mel_times'][0, 0]-.005
        mel_hi = inputs['mel_times'].masked_fill(~inputs['mel_valid'], -1e9).max()+.005
        coverage = ((times[0]>=mel_lo)&(times[0]<=mel_hi)).float()[:, None]
        features = torch.cat((local, song, raw, self.time_features(times[0]), coverage,
                              self.atom_features(values, memory['origin'], device)), -1)
        self._coverage(memory, values, role)
        return self.head_feature(features)

    def forward(self, memory, state):
        device = memory['context'].device
        status = torch.tensor(state['status'], device=device, dtype=torch.long)[None]
        heads = np.asarray(state['head'], np.int64);ends = np.asarray(state['end'], np.int64)
        h = torch.tensor((heads-memory['origin'])/1e6, device=device, dtype=torch.float32)
        e = torch.tensor((ends-memory['origin'])/1e6, device=device, dtype=torch.float32)
        kinds = torch.tensor(state['kind'], device=device, dtype=torch.long).clamp_min(0)
        features = torch.cat((self.time_features(h), self.time_features(e),
            self.atom_features(heads, memory['origin'], device), self.atom_features(ends, memory['origin'], device),
            self.kind_embedding(kinds)), -1)
        x = self.position.weight[None]+self.status(status)+self.event_projection(features)[None]*(status==2)[..., None]
        for layer in self.layers:
            x = layer(x, memory['context'], memory['valid'])
        x = self.norm(x)
        return dict(states=x, active=self.active_read(x).squeeze(-1), memory=memory)

    def head_logits(self, output, slots):
        return self.head_query(output['states'][0, slots]) @ output['memory']['head_keys'].T / math.sqrt(self.c.d)

    def type_logits(self, output, slots, addresses):
        return self.type_read(torch.cat((output['states'][0, slots], output['memory']['head_features'][addresses]), -1))

    def tail_scores(self, output, slot, address, us, level):
        memory=output['memory'];inputs=memory['inputs'];device=output['states'].device
        values=np.asarray(us,np.int64);times=torch.tensor((values-memory['origin'])/1e6,device=device,dtype=torch.float32)[None]
        song=read_grid(memory['song'],memory['song_times'],memory['song_valid'],times,2.)[0]
        raw=read_grid(inputs['mel'],inputs['mel_times'],inputs['mel_valid'],times,.01)[0]
        lo=inputs['mel_times'][0,0]-.005;hi=inputs['mel_times'].masked_fill(~inputs['mel_valid'],-1e9).max()+.005
        cover=((times[0]>=lo)&(times[0]<=hi)).float()[:,None]
        keys=self.tail_key(torch.cat((song,raw,self.time_features(times[0]),cover,self.atom_features(values,memory['origin'],device)),-1))
        query=self.tail_query(torch.cat((output['states'][0,slot],memory['head_features'][address]),-1))
        self._coverage(memory,values,'endpoint_candidates')
        return self.tail_residual(torch.nn.functional.gelu(keys+query+self.tail_level.weight[level])).squeeze(-1)
