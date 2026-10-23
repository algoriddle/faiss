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
float32 Euclidean clustering and makes no speed claim. See
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
with their own communication and call local `update`. CUDA accepts native FP32
and requires finite caller-validated values; metadata is always checked, with
no implicit CUDA value scan. NumPy/Torch CPU validation still checks finiteness.
Local batches above 2²⁴ rows use temporary FP64 sums/counts accumulation before
returning FP32 statistics, requiring additional memory and computation.

Default resident CUDA `train` may replay full batches through an epoch-local
CUDA graph. Capture and instantiation are part of each training call; its private
epoch-owned pool releases capture temporaries after epoch reporting. Explicit
overrides, custom models, tails and very wide accumulation stay eager.
Caller-driven loops remain eager. Capture/execution errors propagate, without retry.
CUDA replay was validated with Torch 2.14; support on older versions is unestablished.

Untouched centroid positions survive mass decay. The obsolete
`recipe["readout_epsilon"]` floor is removed. Revival keeps ordinary
multiplicative splits; a collapsed FP32 pair changes one coordinate to an
adjacent finite value. Distinct operands do not guarantee both children get
rows (a zero donor's subnormal distance can underflow). Final FP32 centroids
are means of the final pass's own assignments; never-assigned centroids retain
their current positions. `obj` and the `train` return remain moving-pass
objectives, rather than scores against the closing means.

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

## Numerical limits

The training objective follows the moving-pass assignments, not the returned
closing centroids. When precision matters, also assess the final centroids on
held-out data in the original coordinates. Validation is caller-controlled;
there is no automatic scan, centering or precision fallback.
