# E-means

Implements e-means, a mini-batch k-means algorithm that updates FP32 centroid
positions and masses with exponential moving averages and revives underused clusters by
splitting active ones. `Emeans` has a Faiss-style `train`/`assign` interface:

```python
from faiss.contrib.e_means import Emeans

km = Emeans(d, k, niter=25)
km.train(x)
distances, labels = km.assign(x)
```

The default statistical batch size and EMA step are selected together; set
`batch_size` and `alpha0` together to override them. NumPy input uses exact
Faiss L2 assignment. PyTorch input stays on its CPU or CUDA device and does not
call Faiss; PyTorch is optional. This reference implementation supports dense
floating Euclidean clustering and makes no speed claim. See
`../../demos/demo_e_means.py` for a DEEP1M example.

For caller-driven batches, supply FP32 initial centroids and the global row
count. `train` uses the same numerical operations with seeded initialization
and shuffling:

```python
km.begin(global_n, init_centroids)
for epoch in range(km.niter):
    for batch in batches(epoch, km.recipe["batch_size"]):
        km.update(km.accumulate(batch))
    km.end_epoch()
centroids = km.finalize()
```

Each epoch has `ceil(global_n / batch_size)` updates, including its final
remainder. The caller owns loading, ordering, transfers and prefetch. Centroids
and masses remain resident. `accumulate` returns model-owned FP32 sums/counts,
an FP64 assignment-objective scalar and a host row count; consume or copy these
before another accumulation. CPU callers can sum all four statistics fields
with their own communication and call local `update`. CUDA requires finite
caller-validated values fitting the chosen format; metadata is always checked, with
no implicit CUDA value scan. NumPy/Torch CPU validation still checks finiteness.

Untouched centroid positions survive mass decay. The obsolete
`recipe["readout_epsilon"]` floor is removed. Revival keeps ordinary
multiplicative splits; a collapsed represented pair changes one coordinate to an
adjacent finite value. Distinct operands do not guarantee both children get
rows (a zero donor's subnormal distance can underflow). Final FP32 centroids
are means of the final pass's own assignments; never-assigned centroids retain
their current positions. `obj` and the `train` return remain moving-pass
objectives, rather than scores against the closing means.

Those final-pass assignments can be kept instead of recomputed. With
`capture_assignments=True`, `train` stores the int64 label of every input row in
`km.assignments` (input order), and `accumulate` returns them as
`stats["labels"]` during the final pass of a caller-driven run:

```python
km = Emeans(d, k, niter=25, capture_assignments=True)
km.train(x)
labels = km.assignments  # each closing centroid is the mean of its rows here
```

No extra pass or search is run. The labels were chosen against centroids that
were still moving, so `assign(x)` against the closing centroids can differ for
some rows; the captured labels are the ones the closing means were built from.

CUDA native Torch accepts FP32, FP16 and BF16. With `quantizer=None`, the
first batch (including a typed empty batch) fixes the native dtype for the run.
FP32 initial centroids do not select that dtype. Norms, dot products and FP32
statistics use the same represented rows; centroid operands follow that format.
FP32 centroid state and final means remain unquantized. Distances refer to the
rounded centroid operands, and native FP32 honors the caller's Torch matmul
settings. FP16/BF16 assignment requires NVIDIA SM80 or later and Torch
`addmm` support for FP32 output.

For explicit batchwise conversion, use
`CastQuantizer(d, torch.float16)` or `CastQuantizer(d, torch.bfloat16)` from
`faiss.contrib.e_means.e_means_quantizer`, passed as `quantizer=q`. Its `train` is a no-op;
`compute_codes` returns native typed Torch rows and `decode` widens them to
FP32. Conversion/decoding check metadata without CUDA value scans. Call
`q.validate(x)` explicitly to check finite/range hazards; unchecked overflow
follows Torch cast semantics. The existing internally generated FP16 split
range guard remains. CPU computes supplied represented floating values in FP32
and requires `quantizer=None`, even though the cast helper can execute on CPU.

`assignment_backend="triton"` explicitly selects optional Triton CUDA assignment
instead of the default native Torch implementation. It requires NVIDIA SM80 or
later, supports the same FP32/FP16/BF16 represented inputs and cast quantizer,
and retains the shared lifecycle/statistics. FP32 dots request IEEE precision.
It uses bounded centroid tiles and split-K partial results without runtime
autotuning or backend fallback. `centroid_chunk_size` applies only to native
Torch; Triton chooses its internal tile geometry. Conservative known-size
parallelism/work-per-partition rules are performance defaults, not hardware
legality or a portable crossover. Padded physical index/label bounds still apply.

GPU callers use `DistributedEmeans(d, k, group=group, transport="auto", ...)`
with an initialized, caller-owned process group. Group and transport are fixed
at construction. `begin(global_n, init_centroids)` takes a same-shape tensor on
each rank's CUDA device and broadcasts its owned FP32 centroid state from group
rank zero. Root values are authoritative; caller tensors are never modified.
The remaining `accumulate`/`update`/`end_epoch`/`finalize` lifecycle is shared
with `Emeans`; distributed `train(x)` is rejected. Centroid state is replicated;
the caller supplies matching recipes/codecs and disjoint rank-local contributions
to each prescribed global batch. Every batch, including the final tail, must
divide evenly by group size; this is checked at `begin` before broadcasting.
Each update checks that local `stats["n"]` equals its prescribed global size
divided by group size. Every rank participates in the same updates and epoch
closes. Local `Emeans.update(stats)` has no group or transport arguments;
`DistributedEmeans` requires an explicit group, not an implicit world group.

Dense transport all-reduces FP32 statistics. Sparse transport packs occupied
IDs, all-gathers and merges in group-rank order. It can synchronize during
packing/size discovery; padded receive storage is approximately
`ranks * max_local_occupied * (d+1) * 4` bytes plus IDs. Automatic selection
uses known batch/table/rank sizes, conservatively bounds padding and amortizes
packing on large payloads; it is not a portable network crossover. Use an
explicit `"dense"` or `"sparse"` override where appropriate, and ensure the
state and transport workspace fit. No batch size is silently reduced.

One scalar FP64 epoch collective sums local moving-pass objective contributions.
Already-global FP64 pass counts and FP32 closing sums are captured once. Equal
row counts can still have different occupied-centroid counts, which sparse
transport discovers separately. Divergent collective schedules remain a caller
error; this API does not recover from them. Revival IDs are
selected by group rank zero and broadcast; every replica applies the same
split. CPU communication remains caller-owned: use NumPy/FP32 `Emeans`,
combine local `sums`, `counts` and objective externally (for example Gloo
`all_reduce` on contiguous buffers made with `torch.from_numpy`), then call local `update`
with the combined statistics and prescribed global `n`. All replicas need
identical initialization, recipe, seed and collective schedules. Do not pass
a CPU group to local `update`; the distributed subclass requires CUDA state. See
`../../demos/demo_e_means_distributed.py` for a small `torchrun` example.

For calibrated E4M3 on qualified Triton CUDA GPUs, use an explicit quantizer:

```python
from faiss.contrib.e_means.e_means_fp8 import Fp8Quantizer
q = Fp8Quantizer(x.shape[1])
q.train(representative_sample)
q.to(x.device)
q.validate(x)  # optional explicit finite/range scan
km = Emeans(x.shape[1], k, quantizer=q, assignment_backend="triton")
km.train(x)  # floating chunks encode during assignment
# Alternatively: km.train(q.compute_codes(x))
centroids = km.centroids  # already decoded FP32; do not decode again
```

FP8 codes with `quantizer=None` define raw numeric code units. FP8 input requires
NVIDIA SM90, SM100 or SM103. Conversion does not clip or scan values: unchecked
out-of-range/nonfinite values can produce NaN codes and invalid results. The
existing internal-centroid range check runs at epoch close and during assign.

Native INT8/UINT8 assignment uses exact INT32 ranking (dimension at most
43,919), with FP32 returned distances, on qualified NVIDIA SM80/SM90/SM100.
An explicit INT8 quantizer returns decoded centroids and distances:

```python
from faiss.contrib.e_means.e_means_int8 import Int8Quantizer

q = Int8Quantizer(d)
q.train(representative_sample)
codes = q.compute_codes(x)  # rounds and clips finite outliers, in bounded chunks
km = Emeans(d, k, quantizer=q, assignment_backend="triton")
km.train(codes)  # matching native codes only; no floating-to-INT8 JIT
```

Without a quantizer, native signed/unsigned bytes retain their numeric units.
Calibration, device placement and validation are explicit; unchecked nonfinite
values have no meaningful encoding result. Use `q.validate(x)` when needed.

CuTe BF16/E4M3 assignment is an explicit optional backend on exactly NVIDIA SM90,
requiring `quack-kernels==0.6.5`, `nvidia-cutlass-dsl==4.7.1` and
`cuda-python>=13,<14`. Supply native BF16 rows with `quantizer=None`, or use
`CastQuantizer(d, torch.bfloat16)` for per-chunk floating conversion; select
`assignment_backend="cute"`. It uses resource-constrained tiled/row-owned
assignment, native FP32 score ranking and no runtime autotuning or fallback.
Bit-equal scores select the lower centroid ID; FP64-only ties can differ.
E4M3 uses native codes or caller-calibrated `Fp8Quantizer` batch conversion,
with decoded FP32 state/statistics and stock exponent-scaled distances. FP8
tensor-core partial accumulation uses FP32 promotion every 128 contraction
coordinates. Ranking is approximate; distances are recomputed from selected
represented operands. Unchecked invalid
JIT inputs have no meaningful result and need not match Torch cast overflow.
Calls on one model share mutable workspace; the caller must order stream switching.

CuTe also accepts native INT8/UINT8 on SM90, with the same 43,919 dimension
bound and caller-owned `Int8Quantizer` workflow above. Select
`assignment_backend="cute"` when training native codes; floating-to-INT8 JIT
is unsupported. Integer scores and lower-ID ties are exact; wide final distances
are reconstructed through INT64 before returning FP32 values.
