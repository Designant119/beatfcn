# Shared Pupu24 plus full KDA4 architecture

This is a numerical scaffold, not a completed chart-writing model.

## Fixed architecture and exact actual tensor counts

- Native Pupu patch embed +24 pretrained layers, d1024,16 heads, preserved
  2D audio RoPE. Strict294-tensor loading,403,141,632 frozen parameters.
- Audio patches AND right-shifted event-group feature tokens traverse the SAME
  QKV, projection, normalization and SwiGLU weights in ALL24 layers, in one
  masked shared pass. There is no independent40M event Transformer.
- Four NEW FULL serial blocks after6/12/18/24: pre-RMSNorm +8head×128 KDA
  residual +pre-RMSNorm +SwiGLU4096 residual. Each17,328,264 trainable params;
  all four69,313,056. This includes50,331,648 new FFN parameters.
- Small new feature/intent/position/selection/probe heads:4,285,229 params.
- **Total476,739,917; trainable73,598,285; frozen403,141,632.**
  BF16 weights953,479,834 bytes; FP32 weights1,906,959,668 bytes.
  These are storage counts, NOT peakVRAM or a4GB feasibility claim.

`reports/parameter_inventory.json` is computed from actual instantiated meta
parameters, not a paper/model-card size label. Base publication revision:
`c0f9345a1de0e61237dec7da32a70049c4ec0fa8`. Local recovered encoder subset
SHA256:`6a97605541ba9d3680f6758e2ae19019736f03e5db14e8bbfbd71746b19304f4`.
Original full-publication-file hash was NOT verified by this recovered subset.
Weights/audio are intentionally not copied into this package.

## KDA definition and source boundary

Independent PyTorch reference follows published Kimi Delta Attention, not
ordinary linear attention: q/k/v projections ->causal depthwise conv4 ->SiLU;
L2 q/k; key-channel decay first; beta delta correction against the decayed
state; q readout with1/sqrt(head_dim); per-head RMSNorm and sigmoid output gate.
Alpha uses low-rank128→1024 gate, softplus, learned per-head exp(A_log), and
per-key dt_bias. Beta is sigmoid, NOT doubled. Value is NOT L2 normalized.
State is FP32[B,H,K,V] (float64 in equation test).

Chosen S0 config: qkv/output/beta/lowrank projections bias-free, conv biases
included, output norm affine, L2 eps1e-6, RMS eps1e-6, A_log initialized0,
dt_bias−3, conv normal std0.5, default PyTorch linear initialization. These are
explicit engineering choices. This is NOT bit-identical to the complete released
Moonshot/FLA module or its initialization. No checkpoint for new KDA is claimed.

Primary source parameterization:
https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/modeling_kimi.py
(MIT; revision verified from model API).
Reference recurrence and optional kernel:
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py
(MIT; moving main reviewed; official PyPI fla-core0.5.2 wheel now pinned,
SHA256-verified and API statically inspected; NOT installed or GPU-tested).
Paper: https://arxiv.org/abs/2510.26692 .

`backend='fla'` calls the actual optional `fla.ops.kda.chunk_kda` kernel with
already-normalized q/k and transformed gates/beta, state_v_first=False. It
never silently switches to generic linear attention. CUDA/FLA forward-backward
parity has NOT been run. Use the verified `requirements-fla.txt` pin and separately pin a compatible
CUDA PyTorch>=2.7/Triton>=3.3/einops environment; run `scripts/check_fla.py`
before using that backend. Its successful T16 and cross-chunk T129 forward/state/backward report is
mandatory, matched to GPU/Torch/CUDA/Triton/BF16 and local KDA source hash. The pinned wheel signature matches the wrapper; runtime kernel/import/math
compatibility remains a pre-GPU gate. Incremental inference cache is explicitly reference-only.

## Visibility and recurrent-state schedule

Order is A (local+global music), R (paired reference feature placeholders), I
(8 learned intent tokens), W (known/control feature placeholders), E (BOS then
completed previous-group attributes). True boolean SDPA edges:

- A reads valid A only, including available future music, never labels/R/W/E.
- R reads R only. Paired reference audio must be encoded in the legal R input
  upstream; that actual reference builder is deferred.
- I reads R/I only. W reads all legal prefix, not E.
- E reads all legal prefix and its own/past shifted positions. Position0=BOS
  predicts group0; positioni embeds groupi−1 and predicts groupi.

At each KDA block FIVE separate shared-weight scans are used, NOT one scan
with a claimed arbitrary mask: A alone; R alone; R→I; A→R→I→W; and
A→R→I→W prefill then BOS/previous-groups. Each route has separate conv and
recurrent state. The scan for each query class outputs only that class's rows.
No reference/intent state ever accepts workspace or target data. Prefix scans
are ordered; bidirectional legal-prefix attention is provided by Pupu SDPA,
not invented as a property of KDA. This duplicates prefix compute and must be
included in later performance estimates.

Per-example valid positions are compacted before recurrence, so masked tokens
advance neither conv nor recurrent state. Batch examples start fresh. Cached
steps retain each layer's prefix/event K/V, valid-key mask and each KDA event
state. Cache reuse checks exact context hash, explicit revision AND parameter
object/version fingerprints. An edited context, changed weights or different
model rejects it; re-prefill required. Caches are inference-only, in-memory,
not a serialized long-term cache API. Do not mutate parameters through `.data`.

Native audio gets native2D RoPE; abstract global summaries and other tokens get
identity RoPE. Event index has learned embeddings and continuous attributes
have an MLP. This positional choice is a scaffold, not trained alignment.

S0 local input is ONE bounded native≤10.24s normalized-mel window; no16s
stitching, center-retain overlap policy or remote retrieval is implemented.
Global music accepts≤600 explicit audio-only feature tokens and remains visible
to events, including future music. Reference/known/global-token prefix≤1536;
local raw patch tokens are counted separately. Excess raises resource_exhausted,
never crops labels. Raw audio has8 frequency tokens per40ms. A tiny selection
readout queries isolated intent from audio only, then averages local scores
across8 frequencies into25Hz cells. It is untrained, not strict CF supervision
or evidence of learned human selection. Upstream semantic provenance is the
caller's responsibility; hidden labels must NEVER be supplied as known features.

## Autograd and caches

Original pretrained parameters are frozen, but ALL shared layer forwards keep
autograd. This propagates to event embeddings and each interleaved KDA. There
is no old lower16 no_grad branch. Even audio layer1 receives trainable modality
embedding here, so first6 activations are NOT reusable across training updates.
H20/H24 also depend on KDA and must not be reused stale. No training activation
cache is implemented. No LoRA in S0a.

## What actually ran

The core CPU suite contains 11 tests, reduced-width d32 BUT24 shared layers and4
full KDA insertions. Tests cover actual full-size meta inventory; independent
matrix-equation forward/backward; conv+recurrent step parity; shared QKV identity
for audio/events; all-new-parameter nonzero finite gradients; forbidden
future/intent Jacobians; hidden-future perturbations; padding/batch reset;
teacher-forced vs cached decoding; context/model/weight invalidation; global
music reachability; strict-loader missing-key rejection; toy save/reload parity.

`reports/real_weight_cpu_probe.json`:actual full476.74M model, strict294-tensor
checkpoint load and verified local SHA256;40ms real local audio excerpt plus
synthetic reference/known/global/previous-group features; BF16 forward/backward
and one explicit SGD update. All trainable modules have finite gradients,
frozen gradients absent, sampled frozen values unchanged. CPU peakRSS≈2.70GB.
This is not a real supervised chart task, a full native-window performance
measurement, GPU benchmark, full frozen-tensor posthash, or audio quality result.
Only a short local technical smoke; training/redistribution rights unverified.

