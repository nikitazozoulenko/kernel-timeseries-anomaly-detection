import numpy as np
import torch
from torch import Tensor
from typing import List, Optional, Dict, Set, Callable, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernels import StaticKernel, TimeSeriesKernel
from kernels import LinearKernel, RBFKernel, PolyKernel
from kernels import TruncSigKernel, SigPDEKernel, StaticIntegralKernel, FlattenedStaticKernel, GlobalAlignmentKernel, ReservoirKernel
from kernels import sigma_gak

################################################################################################## |
## Base class to inherit. Works for any Hilbert space and any inner product given a Gram matrix ## |
################################################################################################## \/

class BaseclassAnomalyScore():
    def __init__(self, 
                 inner_prod_Gram_matrix:Tensor, 
                 SVD_threshold:float = 0.0001,
                 verbose:bool = False,
                 SVD_max_rank:Optional[int] = None,
                ):
        """Class which computes the conformance score or Mahalanobis distance to a 
        given corpus of elements {x_1, ..., x_N} originating from a Hilbert space.

        Args:
            inner_prod_Gram_matrix (Tensor): Gram matrix of shape (N,N) of inner 
                                               products <x_i, x_j>.
            SVD_threshold (float): Sets all eigenvalues of the covariance operator 
                                   below this threshold to be 0.
            verbose (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Only allow 'SVD_max_rank' number of eigenvalues to 
                                be non-zero.
        """
        N,N = inner_prod_Gram_matrix.shape

        #calculate Gram matrix A_{i,j} = <f_i, f_j>. SVD decomposition A= U S U^t
        B = inner_prod_Gram_matrix #<x_i, x_j>
        a = torch.mean(B, dim=0)
        b = torch.mean(a)
        A = ( B - a[None] - a[:, None] + b) / N #<f_i, f_j>

        #SVD decomposition is equal to spectral decomposition
        U, S, Ut = torch.linalg.svd( A )
        M = torch.sum(S > SVD_threshold)
        M = max(M, 1)
        M = min(M, SVD_max_rank) if SVD_max_rank is not None else M
        if verbose:
            print("Covariance operator eigenvalues =", S)
            print("Covariance operator numerical rank =", M)
        U, S = U[:, 0:M], S[0:M] #Shapes (N,M) and (M)


        #calculate matrix E_{i,m} = <x_i, e_m>,  (e_1, ... e_M) ONB of eigenvectors of covariance operator
        E =  (B-a[:None]) @ U / torch.sqrt(N*S[None, :]) #shape (N,M)
        c = torch.mean(E, axis=0) #shape (M)

        #save
        self.E, self.U, self.S, self.c = E, U, S, c


    def _conformance_score(self, 
                           inner_prod_y_en:Tensor, # shape (..., N)
                           return_all_levels:bool,
                           ):
        """Calculates the nearest neighbour variance distance of a new sample 'y' given 
        array of inner products <y, e_n> of eigenvectors of the covariance operator."""
        # d[l,n,m] = <y_l-x_n, e_m>^2 / S_m
        d = inner_prod_y_en[..., None,:]-self.E[:, :] #shape (..., N, M)
        d = d**2 / self.S[None, :] #shape (..., N, M)

        # cumsum[l,n,m] = ||y_l-x_n||^2_{var-norm} at m'th threshold level
        cumsum = torch.cumsum(d, axis=-1) #shape (..., N, M)
        nn_distance, _ = torch.min(cumsum, axis=-2)
        nn_distance = torch.sqrt(nn_distance) #shape (..., M)

        if return_all_levels:
            return nn_distance #shape (..., M)
        else:
            return nn_distance[..., -1] #shape (...)
    

    def _mahalanobis_distance(self,
                              inner_prod_y_en:Tensor, # shape (..., N)
                              return_all_levels:bool,
                              ):
        """Calculates the Mahalanobis distance of a new sample 'y' given
        array of inner products <y, e_n> of eigenvectors of the covariance operator."""
        #d[l,m] = <y_l - xbar, e_m>^2 / S_m
        d = inner_prod_y_en-self.c #shape (..., M)
        d = d**2 / self.S #shape (..., M)

        # cumsum[l,m] = ||y_l-xbar||^2_{var-norm} at m'th threshold level
        sqrt_cumsum = torch.sqrt(torch.cumsum(d, axis=-1)) #shape (..., M)

        if return_all_levels:
            return sqrt_cumsum #shape (..., M)
        else:
            return sqrt_cumsum[..., -1] #shape (...)


    def _anomaly_distance(self, 
                          inner_prod_y_xn : Tensor,
                          method:str = "mahalanobis",
                          return_all_levels:bool = False
                          ):
        """ Returns the anomaly distance of a new sample 'y' with respect to the 
            corpus {x_1, ..., x_N}. Uses either conformance score (nearest neighbour 
            variance distance), or Mahalanobis distance (variance distance to the mean).

        Args:
            inner_prod_y_xn (torch.andrray): Array of shape(..., N) of inner products 
                                    <y_k, x_n> (w.r.t. hilbert space inner product).
            method (str): Either "mahalanobis" or "conformance".
            return_all_levels (bool): If true, returns the anomaly distance at all 
                                      threshold levels.
        
        Returns Tensor: Anomaly distance of shape (...), (2, ...), (..., M), or 
                            (2, ..., M) depending on 'method' and 'return_all_levels'.
        """
        N,M = self.U.shape

        # p_m = <y, e_m>
        p = inner_prod_y_xn - torch.mean(inner_prod_y_xn, axis=-1, keepdims=True) #shape (..., N)
        p = (p @ self.U) / torch.sqrt(N*self.S) #shape (..., M)

        if method == "conformance":
            return self._conformance_score(p, return_all_levels)
        elif method == "mahalanobis":
            return self._mahalanobis_distance(p, return_all_levels)
        elif method == "both":
            return torch.stack((self._conformance_score(p, return_all_levels), 
                            self._mahalanobis_distance(p,return_all_levels)))
        else:
            msg = "Argument 'method' must be in ['mahalanobis', 'conformance', 'both']."
            raise RuntimeError(msg)
    

############################################################################################## |
############################### Kernelized Conformance Score  ################################ |
############################################################################################## \/


class KernelizedAnomalyScore(BaseclassAnomalyScore):
    def __init__(self, 
                corpus:List[Any],
                SVD_threshold:float = 0.0001,
                verbose:bool = False,
                SVD_max_rank:Optional[int] = None,
                ts_kernel:TimeSeriesKernel = GlobalAlignmentKernel()
                ):
        """Callable class which computes the kernelized anomaly score with respect to a corpus of data.

        Args:
            corpus (List): List of elements beloning to the same class.
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            verbose (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
            n_jobs (int): Number of parallel jobs to run in the kernel calculations.
        """
        self.corpus = corpus
        self.ts_kernel = ts_kernel

        #compute kernel Gram matrix
        inner_prod_Gram_matrix = ts_kernel(corpus, corpus)
        super().__init__(inner_prod_Gram_matrix, SVD_threshold, verbose, SVD_max_rank)
    
    def __call__(self, 
                 new_sample:Tensor,
                 method:str = "conformance",
                 ) -> Tensor:
        """ Returns the kernelized anomaly distance with respect to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            new_sample: Time series of shape (T, d).
            method (str): Either "mahalanobis", "conformance", or "both".

        Returns:
            float: Anomaly distance.
        """
        # kernel as inner product
        inner_products = self.ts_kernel(new_sample, self.corpus)
        return self._anomaly_distance(inner_products, method)
    

############################################################################################## |
################################## Anomaly distance in R^d ################################### |
############################################################################################## \/


# NOTE: Can be more efficient to consider covariance matrix X^t X
# instead of the inner product Gram matrix X X^t.
class RdAnomalyScore(BaseclassAnomalyScore):
    def __init__(self, 
                 corpus:Tensor, 
                 SVD_threshold:float = 0.0,
                 verbose:bool = False,
                 SVD_max_rank:Optional[int] = None,
                ):
        """Callable class which computes the conformance score to a given corpus of (finite dimensional) data.

        Args:
            corpus (Tensor): Array of shape (N,d) of d-dimensional feature vectors.
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            verbose (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
        """
        self.corpus = corpus
        inner_prod_Gram_matrix = corpus @ corpus.T  #<x_i, x_j>
        super().__init__(inner_prod_Gram_matrix, SVD_threshold, verbose, SVD_max_rank)
    
    def __call__(self, 
                 y:Tensor,
                 method:str = "conformance"
                 ):
        """ Returns the anomaly distance of 'y' with respect to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            y (Tensor): Feature vector of same dimension as the vectors in the corpus.
            method (str): Either "mahalanobis", "conformance", or "both".
        """
        # euclidean inner product
        inner_prod_y_xn = y @ self.corpus.T
        return self._anomaly_distance(inner_prod_y_xn, method)


############################################
############## Example usage ###############
############################################

if __name__ == "__main__":
    import time
    from tslearn.datasets import UCR_UEA_datasets
    from models.stream_transforms import normalize_streams

    #get corpus, in-sample, and out-of-sample
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("BasicMotions")
    X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
    normal_class = y_train[0]
    corpus = X_train[y_train == normal_class]
    corpus, test = normalize_streams(corpus, X_test)
    in_sample = test[y_test == normal_class][0]
    out_sample = test[y_test != normal_class][0]
    N, T, d = corpus.shape

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
        print("{}:".format(name))
        start = time.perf_counter()
        print("Anomaly distance for new sample, same distribution:     ", scorer(in_sample))
        print("Anomaly distance for new sample, different distribution:", scorer(out_sample))
        print("Time taken: {}\n\n".format(time.perf_counter()-start))

    anomaly_test("Flattened", flattened_scorer, in_sample.reshape(1, T*d), out_sample.reshape(1, T*d))
    anomaly_test("Integral Kernel", int_scorer, in_sample, out_sample)
    anomaly_test("Truncated Signature", sig_scorer, in_sample, out_sample)
    anomaly_test("GAK Kernel", gak_scorer, in_sample, out_sample) # takes a couple of ms to jit dynamic programming solution