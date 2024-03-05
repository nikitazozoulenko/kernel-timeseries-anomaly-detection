import numpy as np
import iisignature
import sigkernel
import torch
from typing import List, Optional, Dict, Set, Callable, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.kernels import pairwise_kernel_gram, sig_kernel_gram, integral_kernel_gram
from models.kernels import linear_kernel_gram, poly_kernel_gram, rbf_kernel_gram


################################################################################################## |
## Base class to inherit. Works for any Hilbert space and any inner product given a Gram matrix ## |
################################################################################################## \/

class BaseclassConformanceScore():
    def __init__(self, 
                 inner_prod_Gram_matrix:np.ndarray, 
                 SVD_threshold:float = 0.0001,
                 verbose:bool = False,
                 SVD_max_rank:Optional[int] = None,
                ):
        """Class which computes the conformance score or Mahalanobis distance to a 
        given corpus of elements {x_1, ..., x_N} originating from a Hilbert space.

        Args:
            inner_prod_Gram_matrix (np.ndarray): Gram matrix of shape (N,N) of inner 
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
        a = np.mean(B, axis=0)
        b = np.mean(a)
        A = ( (B - np.expand_dims(a, axis=0)) - np.expand_dims(a, axis=1) + b) / N #<f_i, f_j>

        #SVD decomposition is equal to spectral decomposition
        U, S, Ut = np.linalg.svd( A )
        M = np.sum(S > SVD_threshold)
        M = max(M, 1)
        M = min(M, SVD_max_rank) if SVD_max_rank is not None else M
        if verbose:
            print("Covariance operator eigenvalues =", S)
            print("Covariance operator numerical rank =", M)
        U, S = U[:, 0:M], S[0:M] #Shapes (N,M) and (M)


        #calculate matrix E_{i,m} = <x_i, e_m>,  (e_1, ... e_M) ONB of eigenvectors of covariance operator
        E =  (B-np.expand_dims(a, axis=1)) @ U / np.sqrt(N*S[None, :]) #shape (N,M)
        c = np.mean(E, axis=0) #shape (M)

        #save
        self.E, self.U, self.S, self.c = E, U, S, c


    def _conformance_score(self, 
                           inner_prod_y_en:np.ndarray, # shape (..., N)
                           return_all_levels:bool,
                           ):
        """Calculates the nearest neighbour variance distance of a new sample 'y' given 
        array of inner products <y, e_n> of eigenvectors of the covariance operator."""
        # d[l,n,m] = <y_l-x_n, e_m>^2 / S_m
        d = inner_prod_y_en[..., None,:]-self.E[:, :] #shape (..., N, M)
        d = d**2 / self.S[None, :] #shape (..., N, M)

        # cumsum[l,n,m] = ||y_l-x_n||^2_{var-norm} at m'th threshold level
        cumsum = np.cumsum(d, axis=-1) #shape (..., N, M)
        nn_distance = np.sqrt(np.min(cumsum, axis=-2)) #shape (..., M)

        if return_all_levels:
            return nn_distance #shape (..., M)
        else:
            return nn_distance[..., -1] #shape (...)
    

    def _mahalanobis_distance(self,
                              inner_prod_y_en:np.ndarray, # shape (..., N)
                              return_all_levels:bool,
                              ):
        """Calculates the Mahalanobis distance of a new sample 'y' given
        array of inner products <y, e_n> of eigenvectors of the covariance operator."""
        #d[l,m] = <y_l - xbar, e_m>^2 / S_m
        d = inner_prod_y_en-self.c #shape (..., M)
        d = d**2 / self.S #shape (..., M)

        # cumsum[l,m] = ||y_l-xbar||^2_{var-norm} at m'th threshold level
        sqrt_cumsum = np.sqrt(np.cumsum(d, axis=-1)) #shape (..., M)

        if return_all_levels:
            return sqrt_cumsum #shape (..., M)
        else:
            return sqrt_cumsum[..., -1] #shape (...)


    def _anomaly_distance(self, 
                          inner_prod_y_xn : np.ndarray,
                          method:str = "mahalanobis",
                          return_all_levels:bool = False
                          ):
        """ Returns the anomaly distance of a new sample 'y' with respect to the 
            corpus {x_1, ..., x_N}. Uses either conformance score (nearest neighbour 
            variance distance), or Mahalanobis distance (variance distance to the mean).

        Args:
            inner_prod_y_xn (np.andrray): Array of shape(..., N) of inner products 
                                    <y_k, x_n> (w.r.t. hilbert space inner product)
            method (str): Either "mahalanobis" or "conformance".
            return_all_levels (bool): If true, returns the anomaly distance at all 
                                      threshold levels.
        
        Returns np.ndarray: Anomaly distance of shape (...), (2, ...), (..., M), or 
                            (2, ..., M) depending on 'method' and 'return_all_levels'.
        """
        N,M = self.U.shape

        # p_m = <y, e_m>
        p = inner_prod_y_xn - np.mean(inner_prod_y_xn, axis=-1, keepdims=True) #shape (..., N)
        p = (p @ self.U) / np.sqrt(N*self.S) #shape (..., M)

        if method == "conformance":
            return self._conformance_score(p, return_all_levels)
        elif method == "mahalanobis":
            return self._mahalanobis_distance(p, return_all_levels)
        elif method == "both":
            return np.array((self._conformance_score(p, return_all_levels), 
                            self._mahalanobis_distance(p,return_all_levels)))
        else:
            msg = "Argument 'method' must be in ['mahalanobis', 'conformance', 'both']."
            raise RuntimeError(msg)
        

############################################################################################## |
################################## Anomaly distance in R^d ################################### |
############################################################################################## \/


# Note: Can be more efficient to consider covariance matrix X^t X 
# instead of the inner product Gram matrix X X^t.
class RnConformanceScore(BaseclassConformanceScore):
    def __init__(self, 
                 corpus:np.ndarray, 
                 SVD_threshold:float = 0.0,
                 verbose:bool = False,
                 SVD_max_rank:Optional[int] = None,
                ):
        """Callable class which computes the conformance score to a given corpus of (finite dimensional) data.

        Args:
            corpus (np.ndarray): Array of shape (N,d) of d-dimensional feature vectors.
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            verbose (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
        """
        self.corpus = corpus
        inner_prod_Gram_matrix = corpus @ corpus.T  #<x_i, x_j>
        super().__init__(inner_prod_Gram_matrix, SVD_threshold, verbose, SVD_max_rank)
    
    def __call__(self, 
                 y:np.ndarray,
                 method:str = "conformance"
                 ):
        """ Returns the anomaly distance of 'y' with respect to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            y (np.ndarray): Feature vector of same dimension as the vectors in the corpus.
            method (str): Either "mahalanobis", "conformance", or "both".
        """
        # euclidean inner product
        inner_prod_y_xn = y @ self.corpus.T
        return self._anomaly_distance(inner_prod_y_xn, method)
    

############################################################################################## |
############ Truncated Signature Anomaly Distance for d-dimensional Time Series ############## |
############################################################################################## \/


class TruncSigConformanceScore(BaseclassConformanceScore):
    def __init__(self, 
                 corpus:np.ndarray, 
                 SVD_threshold:float = 0.0,
                 verbose:bool = False,
                 SVD_max_rank:Optional[int] = None,
                 order:int = 5,
                 static_kernel_gram = linear_kernel_gram,
                ):
        """Callable class which computes the anomaly distance with respect to a corpus of 
        variable length, d-dimensional time series, via truncated signature features (via 
        the truncated signature kernel).

        Args:
            corpus (np.ndarray): Array of shape (N,T,d) of d-dimensional equal length time series.
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            verbose (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
            order (int): Order of the truncated signature transform.
            static_kernel_gram: A function which takes two ndarrays of ndim=3 and returns the
                                kernel Gram matrix, such as 'rbf_kernel_gram' or 'linear_kernel_gram'.
        """
        self.corpus = corpus
        self.order = order
        self.static_kernel_gram = static_kernel_gram
        inner_prod_Gram_matrix = sig_kernel_gram(corpus, corpus, order, static_kernel_gram)
        super().__init__(inner_prod_Gram_matrix, SVD_threshold, verbose, SVD_max_rank)
    

    def __call__(self, 
                 stream:np.ndarray,
                 method:str = "conformance"
                 ):
        """ Returns the anomaly distance of 'stream' with respect to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            stream (np.ndarray): Time series of shape (T,d) or (N, T, d).
            method (str): Either "mahalanobis", "conformance", or "both".
        """
        #expand dim if necessary
        if stream.ndim == 2:
            stream = np.expand_dims(stream, axis=0)

        # euclidean inner product
        inner_prod_y_xn = sig_kernel_gram(stream, self.corpus, self.order, self.static_kernel_gram)
        return self._anomaly_distance(inner_prod_y_xn, method)
    


############################################################################################## |
####################### Integral class kernel w.r.t static kernel ############################ |
############################################################################################## \/


class IntegralConformanceScore(BaseclassConformanceScore):
    def __init__(self, 
                 corpus:np.ndarray, 
                 SVD_threshold:float = 0.0,
                 verbose:bool = False,
                 SVD_max_rank:Optional[int] = None,
                 static_kernel_gram = linear_kernel_gram,
                ):
        """Callable class which computes the anomaly distance with respect to a corpus of 
        variable length, d-dimensional time series, via integral class kernels w.r.t. a static 
        kernel with 'diag' argument.

        Args:
            corpus (np.ndarray): Array of shape (N,T,d) of d-dimensional equal length time series.
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            verbose (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
            static_kernel_gram: A function which takes two ndarrays of ndim=3 and returns the
                                kernel Gram matrix, such as 'rbf_kernel_gram' or 'linear_kernel_gram'
                                with a 'diag' argument.
        """
        self.corpus = corpus
        self.static_kernel_gram = static_kernel_gram
        inner_prod_Gram_matrix = integral_kernel_gram(corpus, corpus, static_kernel_gram)
        super().__init__(inner_prod_Gram_matrix, SVD_threshold, verbose, SVD_max_rank)
    

    def __call__(self, 
                 stream:np.ndarray,
                 method:str = "conformance"
                 ):
        """ Returns the anomaly distance of 'stream' with respect to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            stream (np.ndarray): Time series of shape (T,d) or (N, T, d).
            method (str): Either "mahalanobis", "conformance", or "both".
        """
        #expand dim if necessary
        if stream.ndim == 2:
            stream = np.expand_dims(stream, axis=0)

        # euclidean inner product
        inner_prod_y_xn = integral_kernel_gram(stream, self.corpus, self.static_kernel_gram)
        return self._anomaly_distance(inner_prod_y_xn, method)


############################################################################################## |
############################### Kernelized Conformance Score  ################################ |
############################################################################################## \/


class KernelizedConformanceScore(BaseclassConformanceScore):
    def __init__(self, 
                corpus:List[Any],
                kernel:Callable,
                SVD_threshold:float = 0.0,
                verbose:bool = False,
                SVD_max_rank:Optional[int] = None,
                n_jobs:int = 1,
                ):
        """Callable class which computes the kernelized anomaly score with respect to a corpus of data.

        Args:
            corpus (List): List of elements beloning to the same class.
            kernel (Callable): A kernel function which takes two elements and returns a non-negative scalar.
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            verbose (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
            n_jobs (int): Number of parallel jobs to run in the kernel calculations.
        """
        self.corpus = corpus
        self.kernel = kernel
        self.n_jobs = n_jobs

        #compute kernel Gram matrix
        inner_prod_Gram_matrix = pairwise_kernel_gram(corpus, corpus, kernel, 
                        sym=True, n_jobs=n_jobs)

        super().__init__(inner_prod_Gram_matrix, SVD_threshold, verbose, SVD_max_rank)
    
    def __call__(self, 
                 new_sample:np.ndarray,
                 method:str = "conformance",
                 ) -> np.ndarray:
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
        inner_products = pairwise_kernel_gram([new_sample], self.corpus, self.kernel, 
                            sym=False, n_jobs=self.n_jobs)
        inner_products = np.array(inner_products)
        return self._anomaly_distance(inner_products, method)
    


if __name__ == "__main__":
    import time
    from tslearn.datasets import UCR_UEA_datasets
    from tslearn.metrics import gak
    from models.normalize_streams import normalize_streams

    #get corpus, in-sample, and out-of-sample
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("BasicMotions")
    normal_class = y_train[0]
    corpus = X_train[y_train == normal_class]
    corpus, test = normalize_streams(corpus, X_test)
    in_sample = test[y_test == normal_class][0]
    out_sample = test[y_test != normal_class][0]
    N, T, d = corpus.shape

    #get 3 different anomaly distance scorers
    poly_ker = lambda x, y, diag : poly_kernel_gram(x, y, 3, 1, diag)
    rbf_ker = lambda x, y : rbf_kernel_gram(x, y, 0.01)
    gak_ker = lambda x, y : gak(x, y, 30)
    flattened_scorer = RnConformanceScore(corpus.reshape(N, T*d), SVD_threshold=0.0001)
    int_scorer = IntegralConformanceScore(corpus, SVD_threshold=0.0001, static_kernel_gram=poly_ker)
    sig_scorer = TruncSigConformanceScore(corpus, order=5, SVD_threshold=0.0001, static_kernel_gram=rbf_ker)
    kernel_scorer = KernelizedConformanceScore(corpus, gak_ker, SVD_threshold=0.0001)

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
    anomaly_test("GAK Kernel", kernel_scorer, in_sample, out_sample)