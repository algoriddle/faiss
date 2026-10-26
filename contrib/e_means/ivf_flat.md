# Optional IVF assignment — full original restoration candidate

This restores Matthijs D127037745/version455068101's feature surface on the
current centroid-owner lifecycle. **The original grouped call, output precision
and planner are retained by owner decision**, including their numerical limits.
There is no operand widening, fallback or replacement kernel. Earlier
default-only P11 qualification does not qualify the restored options.

## Model and imports

```python
from faiss.contrib.e_means import EmeansShardedCentroidsIVF

model = EmeansShardedCentroidsIVF(
    d, global_k, group=group, nlist=256, nprobe=4,
    coarse_niter=10, search_workspace_bytes=512 << 20)
model.begin(global_n, local_initial_centroids)
# Existing accumulate/update/end_epoch/finalize lifecycle.
```

The same class is available from sharded_centroids.py and
sharded_centroids_ivf.py. Package exports of both sharder classes are lazy;
ordinary package import adds no mandatory Torch dependency. Accessing either
class requires Torch. Caller-owned groups, CUDA centroid shards, equal rank
contributions and current base validation/lifecycle rules remain.

Coarse training preserves original seeds/sample membership, FP16 Triton
assignment and distributed reduction. Coarse centroids are trained once;
local lists are rebuilt on the first positive transfer of each frozen logical
batch and shared by subsequent transfers. Global occupancy is reduced in INT64.
Each owner routes/searches locally, exactly as the original: no extra probe-ID
broadcast. Base score/global-ID winner reductions retain lowest-global-ID ties.

The original accepted inputs reach inherited current preparation rather than
an added FP32-only guard. Other representations/backend selections remain
subject to inherited validation; acceptance is not qualification of every
encoded composition. Original IVF Torch/Triton assignment overrides base
assignment preparation, so a base backend name does not identify fine search.
Quantizers remain rejected by the sharder. Spherical mode is rejected before
collectives because this score is not the current spherical metric.

Current base EMA, sums/counts, SSE, integer/wide-statistics and optional frozen
closing remain. No obsolete accumulation/closing override is transplanted.
Finalize retains the original IVF cache; reset/new begin clear it. Recipe
additions are the original index_factory/nprobe. Revival retains FP16 separation.

## Index options and lower-level APIs

```python
from faiss.contrib.e_means.ivf_flat import IndexIVFFlat

index = IndexIVFFlat(coarse_centroids, nprobe=4, query_chunk=2048,
                     group_size=256, chunk_budget=1 << 26,
                     search_workspace_bytes=512 << 20)
index.add(candidates)
scores, ids = index.search_1nn(queries)
```

| Original mutable attribute | Behavior |
|---|---|
| implem=4 | <2 full direct query operation; >=2 query chunks. <3 native operands; >=3 selected reduced precision. >=4 plus reduced precision stores only reduced candidates; otherwise native/dual storage. |
| dtype="fp16" | "bf16" selects BF16; other values follow the original FP16 comparison. |
| lowp_scope="all" | Other values keep coarse exact helpers native, fine helpers use selected precision. |

These are attributes, not a new constructor/dispatch API. Coarse norms are
cached at construction under default policy; selector mutation does not refresh
them. Add rebuilds fine storage/norms. CUDA candidate-to-coarse assignment always
uses original FP16 Triton; CPU follows coarse exact-helper policy. BF16 fine mode
does not imply BF16 coarse membership on CUDA.

Search_1nn always uses padded search. The original required grouped argument of
search_1nn_batched(x, grouped, coarse_idx=None) is restored: False padded BMM,
True ragged grouped MM. Implem5 does not automatically switch this argument.
Grouped planning pads candidates to multiples16 and retains ragged queries and
INT32 offsets. List_chunks(ql, workspace_bytes=None, grouped=False) preserves
budget estimates, native/reduced item size, lexsort/grouping and hot-list splits.

Exact1nn/top-k helpers, _mm_dtype, _as_computed, _sq_norms, _lowp_operands,
_assign_to_coarse, _timed and _chunk_body retain original signatures/hooks.
Supplied norms/probes keep original semantics. Coarse_quantize converts probes
to query device/INT64; no new exact dtype/device requirement or value scan.

## Precision, ownership and limits

Original reduced MM/BMM output was rounded/overflowed before a later float()
conversion. FP16/BF16 MM/BMM now produce FP32 output directly on CUDA, or dot
FP32-decoded represented operands on CPU. Native operations preserve their
original computation and subsequent float conversion. This is a deliberate
numerical correction, not bit equivalence to inherited reduced output.
Query norms remain omitted; candidate norms follow original stage policy.
Limited probing remains approximate. No normalization/clipping is added.

**Grouped MM remains torch._grouped_mm(...).float().** Root's actual H100 probe
on Torch2.14.0+cu130 executes FP16/BF16/nativeFP32 defaults and nativeFP32 with
explicit FP32 output. Only FP16/BF16 with differing FP32 output reject the call.
The installed Python Meta checks predicted CUDA>=13.3 for FP16 and rejected
nativeFP32; those predictions do not describe this actual eager CUDA path.
No feature guard/skip is based on them. Constant-oracle capability does not
qualify arbitrary ragged selection, capture or performance.

A saved D1024 witness with queries31.75 and candidates31.875/31.75 has an FP64
direct squared-distance margin16. Original FP16 grouped output overflows to
Inf; original BF16 output remains finite but rounds the dots enough to reverse
the ordering. Both select the wrong candidate for all32 queries. These are
documented original precision limits: converting the output to float does not
recover lost precision. The grouped path is not an FP32-output dot guarantee.
Ordinary MM/BMM retain the separate accepted FP32-output correction above.

Rounding operands to the selected low precision and then widening them before
the same grouped primitive fixed that bounded witness. It is not part of this
restoration. A separate cast-inclusive64-group inner-call diagnostic measured
BF16 about27x slower and FP16 about11.5% slower with widening. Those measurements
are not training timings or whole-process memory bounds; optimization remains
separate from the faithfully retained feature.

Original float().contiguous() coarse conversion may alias contiguous FP32
caller storage. Coarse inputs must stay fixed; no clone/cache refresh is added.
Autograd observability remains original, without a new differentiability
promise. Caller orders rebuild/search streams; mutable index is not concurrent-safe.

Returned IDs are INT32 original row IDs; prefixes/group offsets must fit signed
INT32. Exact narrowing guards reject counts above INT32_MAX. Empty/no-local-hit
search returns (+inf,-1). Stable sorting/lowest-ID merge and per-owner ndis /
coarse_ndis accounting remain original.

Fine workspace limits original planner estimates, not whole-process storage.
Native/dual/reduced candidate storage, query representations, sorting/coarse
scores, host metadata, reductions, allocator and base statistics are separate.
CPU reduced-MM/BMM correction adds FP32 decoding temporaries. Positive real
budgets and original mutable None/zero planner semantics remain; insufficient
one-candidate workspace raises.

Tests/torch_test_e_means_ivf.py extends existing meaningful checks for missing
features/concrete bugs. CPU checks do not establish CUDA/grouped/capture,
every representation/backend, quality or throughput. Unsupported/unexecuted
branches stay unqualified. Full original-to-final reviews and Root-owned
qualification follow the frozen candidate.
