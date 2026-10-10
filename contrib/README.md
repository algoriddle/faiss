
# The contrib modules

The contrib directory contains helper modules for Faiss for various tasks.

## Code structure

The contrib directory gets compiled in the module faiss.contrib.
Note that although some of the modules may depend on additional modules (eg. GPU Faiss, pytorch, hdf5), they are not necessarily compiled in to avoid adding dependencies. It is the user's responsibility to provide them.

## List of contrib modules

### rpc.py

A very simple Remote Procedure Call library, where function parameters and results are pickled, for use with client_server.py

### client_server.py

The server handles requests to a Faiss index. The client calls the remote index.
This is mainly to shard datasets over several machines, see [Distributed index](https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM#distributed-index)

### ondisk.py

Encloses the main logic to merge indexes into an on-disk index.
See [On-disk storage](https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM#on-disk-storage)

### exhaustive_search.py

Computes the ground-truth search results for a dataset that possibly does not fit in RAM. Uses GPU if available.
Tested in `tests/test_contrib.TestComputeGT`

### torch_utils.py

Interoperability functions for pytorch and Faiss: Importing this will allow pytorch Tensors (CPU or GPU) to be used as arguments to Faiss indexes and other functions. Torch GPU tensors can only be used with Faiss GPU indexes. If this is imported with a package that supports Faiss GPU, the necessary stream synchronization with the current pytorch stream will be automatically performed.

Numpy ndarrays can continue to be used in the Faiss python interface after importing this file. All arguments must be uniformly either numpy ndarrays or Torch tensors; no mixing is allowed.

Tested in `tests/test_contrib_torch.py` (CPU) and `gpu/test/test_contrib_torch_gpu.py` (GPU).

### inspect_tools.py

Functions to inspect C++ objects wrapped by SWIG. Most often this just means reading
fields and converting them to the proper python array.

### ivf_tools.py

A few functions to override the coarse quantizer in IVF, providing additional flexibility for assignment.

### datasets.py

(may require h5py)

Definition of how to access data for some standard datasets.

### factory_tools.py

Functions related to factory strings.

### evaluation.py

A few non-trivial evaluation functions for search results

### clustering.py

Contains:

- a Python implementation of kmeans, that can be used for special datatypes (eg. sparse matrices).

- a 2-level clustering routine and a function that can apply it to train an IndexIVF

### e_means.py

Implements e-means, a mini-batch k-means algorithm that updates cluster counts
and sums with exponential moving averages and revives underused clusters by
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
call Faiss; PyTorch is optional. PyTorch assignment uses FP32 by default. On
supported CUDA devices, `assignment_precision` can round assignment operands
to FP16 or BF16 while centroid state, statistics, scores, and returned
distances remain FP32. The precision is always selected explicitly. No
performance claim is made here. See `../demos/demo_e_means.py` for a DEEP1M
example.

### big_batch_search.py

Search IVF indexes with one centroid after another. Useful for large
databases that do not fit in RAM *and* a large number of queries.
