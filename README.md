# Welcome!

This Github repo contains a PyTorch implementation of Algorithm 1-3 from the paper ["Variance Norms for Kernelized Anomaly Detection"](https://arxiv.org/abs/2407.11873), on anomaly detection in infinite dimensional Banach and Hilbert spaces.

## Time series kernels
In addition to implementing the (kernelized) Mahalanobis distance to the mean, and the (kernelized) nearest neighbour Mahalanobis distance (also known as the Conformance score), we include efficient GPU-supported implementations of the following time series kernels.

* Integral-class Kernels (linear time warping kernels)
* The Global Alignment Kernel [[1]](https://arxiv.org/abs/cs/0610033)
* The Volterra Reservoir Kernel [[2]](https://arxiv.org/abs/2212.14641)
* The Truncated Signature Kernel [[3]](https://jmlr.org/papers/v20/16-314.html)
* The PDE Signature Kernel [[4]](https://arxiv.org/abs/2006.14794) (implemented as a wrapper to the [sigker](https://github.com/crispitagorico/sigkernel) library)
* Randomized Signatures [[5]](https://arxiv.org/abs/2201.02441)

## Example Usage


```python
import torch
from tslearn.datasets import UCR_UEA_datasets
from models.stream_transforms import normalize_streams
from models.kernels import LinearKernel, PolyKernel, RBFKernel, sigma_gak
from models.kernels import FlattenedStaticKernel, StaticIntegralKernel, TruncSigKernel, GlobalAlignmentKernel
from models.anomaly_distance import KernelizedAnomalyScore

#get corpus, in-sample, and out-of-sample
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("BasicMotions")
X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
normal_class = y_train[0]
corpus = X_train[y_train == normal_class]
corpus, test = normalize_streams(corpus, X_test)
in_sample = test[y_test == normal_class][0]
out_sample = test[y_test != normal_class][0]
N, T, d = corpus.shape
alpha = 0.00001

#create the time series kernel objects
flat = FlattenedStaticKernel(LinearKernel())
integral = StaticIntegralKernel(PolyKernel())
sig = TruncSigKernel(trunc_level=5, static_kernel=RBFKernel(sigma_gak(corpus)))
gak = GlobalAlignmentKernel(static_kernel=RBFKernel(sigma_gak(corpus)))

#get the anomaly detection objects
flattened_scorer = KernelizedAnomalyScore(corpus, ts_kernel=flat)
int_scorer = KernelizedAnomalyScore(corpus, ts_kernel=integral)
sig_scorer = KernelizedAnomalyScore(corpus, ts_kernel=sig)
gak_scorer = KernelizedAnomalyScore(corpus, ts_kernel=gak)

#test the anomaly distances
def anomaly_test(name, scorer, in_sample, out_sample):
    print(f"{name}:")
    print("Anomaly distance for new sample, same distribution:     ", scorer(in_sample, alpha))
    print("Anomaly distance for new sample, different distribution:", scorer(out_sample, alpha))
    print()

anomaly_test("Flattened", flattened_scorer, in_sample.reshape(1, T*d), out_sample.reshape(1, T*d))
anomaly_test("Integral Kernel", int_scorer, in_sample, out_sample)
anomaly_test("Truncated Signature", sig_scorer, in_sample, out_sample)
anomaly_test("GAK Kernel", gak_scorer, in_sample, out_sample)
```
