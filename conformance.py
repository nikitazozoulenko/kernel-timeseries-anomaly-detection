import numpy as np
import iisignature
from tqdm import tqdm
from joblib import Memory, Parallel, delayed
import sigkernel
import torch

from typing import List, Optional, Dict, Set, Callable

from signature import streams_to_sigs, transform_stream


def pairwise_kernel_gram(X:List, 
                         Y:List, 
                         kernel:Callable, 
                         sym:bool = False, 
                         n_jobs:int = 1, 
                         disable_tqdm:bool = False,):
    """Calculates the kernel Gram matrix k(X_i, Y_j) of two collections X and Y
    using joblib.Parallel for parallelization."""
    #Create indices to loop over
    N, M = len(X), len(Y)
    if sym:
        indices = np.stack(np.triu_indices(N)).T #triangular pairs
    else:
        indices = np.stack(np.meshgrid(np.arange(N), np.arange(M))).T.reshape(-1,2)

    #Calculate kernel Gram matrix
    inner_products = Parallel(n_jobs=n_jobs)(
        delayed(kernel)(X[i], Y[j]) 
        for i,j in tqdm(indices, disable = disable_tqdm, desc="Kernel Gram Matrix"))

    #Populate matrix
    inner_prod_Gram_matrix = np.zeros((N,M), dtype=np.float64)
    for (i,j), val in zip(indices, inner_products):
        inner_prod_Gram_matrix[i,j] = val
        if sym:
            inner_prod_Gram_matrix[j,i] = val

    return inner_prod_Gram_matrix


def stream_to_torch(
        raw_stream:np.ndarray, 
        stream_transforms:List[str] = [],
        ):
    """Transforms a raw stream of shape (T, d) into a torch tensor and applies 
    stream transformations."""
    stream = transform_stream(raw_stream, stream_transforms)
    N_i, new_d = stream.shape
    tensor = torch.from_numpy(stream).reshape(1, N_i, new_d).detach()
    return tensor


############################################################################################## |
########## Base class to inherit. Works for any Hilbert space and any inner product ########## |
############################################################################################## \/


class BaseclassConformanceScore():
    def __init__(self, 
                 inner_prod_Gram_matrix:np.array, 
                 SVD_threshold:float = 0.01,
                 print_rank:bool = True,
                 SVD_max_rank:Optional[int] = None,
                ):
        """Class which computes the conformance score or Mahalanobis distance to a given 
           corpus of elements {x_1, ..., x_N} originating from a separable Hilbert space.

        Args:
            inner_prod_Gram_matrix (np.array): Gram matrix of shape (N,N) of inner products <x_i, x_j>.
            SVD_threshold (float): Sets all eigenvalues of the covariance operator below this threshold to be 0.
            print_rank (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Only allow 'SVD_max_rank' number of eigenvalues to be non-zero.
        """
        N,N = inner_prod_Gram_matrix.shape

        #calculate Gram matrix A_{i,j} = <f_i, f_j>. SVD decomposition A= U S U^t
        B = inner_prod_Gram_matrix #<x_i, x_j>
        a = np.mean(B, axis=0)
        b = np.mean(a, axis=0)
        A = ( (B - np.expand_dims(a, axis=0)) - np.expand_dims(a, axis=1) + b) / N #<f_i, f_j>

        #SVD decomposition is equal to spectral decomposition
        U, S, Ut = np.linalg.svd( A )
        M = np.sum(S > SVD_threshold)
        M = min(M, SVD_max_rank) if SVD_max_rank is not None else M 
        U, S = U[:, 0:M], S[0:M]
        if print_rank:
            print("Covariance operator numerical rank = {}".format(M))

        #calculate matrix E_{i,m} = <x_i, e_m>,  (e_1, ... e_M) ONB of eigenvectors of covariance operator
        E =  (B-np.expand_dims(a, axis=1)) @ U / np.sqrt(N*S)
        c = np.mean(E, axis=0)

        #save
        self.E, self.U, self.S, self.c = E, U, S, c


    def _conformance_score(self, 
                           inner_prod_y_en:np.array
                           ):
        """Calculates the nearest neighbour variance distance of a new sample 'y' given 
        array of inner products <y, e_n> of eigenvectors of the covariance operator."""
        #d_n = ||y-x_n||^2_{var-norm}
        d = np.sum((inner_prod_y_en-self.E)**2 / self.S, axis=1)
        nn_distance = np.sqrt(np.min(d))
        return nn_distance
    

    def _mahalanobis_distance(self,
                              inner_prod_y_en:np.array
                              ):
        """Calculates the Mahalanobis distance of a new sample 'y' given
        array of inner products <y, e_n> of eigenvectors of the covariance operator."""
        #d = ||y- xbar||_{var-norm}
        mahalanobis = np.sqrt(np.sum( (inner_prod_y_en-self.c)**2 / self.S ))
        return mahalanobis


    def _anomaly_distance(self, 
                          inner_prod_y_xn : np.array,
                          method:str = "conformance"
                          ):
        """ Returns the anomaly distance of a new sample 'y' with respect to the corpus {x_1, ..., x_N}.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            inner_prod_y_xn (np.array): Array of size N of inner products <x_n, y>.
            method (str): Either "mahalanobis" or "conformance".
        """
        N,M = self.U.shape

        # p_m = <y, e_m>
        p = inner_prod_y_xn - np.mean(inner_prod_y_xn)
        p = (p @ self.U) / np.sqrt(N*self.S)
        
        if method == "conformance":
            return self._conformance_score(p)
        elif method == "mahalanobis":
            return self._mahalanobis_distance(p)
        elif method == "both":
            return self._conformance_score(p), self._mahalanobis_distance(p)
        else:
            raise RuntimeError("Argument 'method' must be in ['mahalanobis', 'conformance'].")


############################################################################################## |
################################## Anomaly distance in R^d ################################### |
############################################################################################## \/


class RnConformanceScore(BaseclassConformanceScore):
    def __init__(self, 
                 corpus:np.array, 
                 SVD_threshold:float = 0.0,
                 print_rank:bool = True,
                 SVD_max_rank:Optional[int] = None,
                ):
        """Callable class which computes the conformance score to a given corpus of (finite dimensional) data.

        Args:
            corpus (np.array): Array of shape (N,d) of d-dimensional feature vectors.
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            print_rank (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
        """
        self.corpus = corpus
        inner_prod_Gram_matrix = corpus @ corpus.T  #<x_i, x_j>
        super().__init__(inner_prod_Gram_matrix, SVD_threshold, print_rank, SVD_max_rank)
    
    def __call__(self, 
                 y:np.array,
                 method:str = "conformance"
                 ):
        """ Returns the anomaly distance of 'y' with respect to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            y (np.array): Feature vector of same dimension as the vectors in the corpus.
            method (str): Either "mahalanobis", "conformance", or "both".
        """
        # euclidean inner product
        inner_prod_y_xn = y @ self.corpus.T
        return self._anomaly_distance(inner_prod_y_xn, method)
    

############################################################################################## |
############ Truncated Signature Anomaly Distance for d-dimensional Time Series ############## |
############################################################################################## \/


class TruncSigConformanceScore():
    def __init__(self, 
                stream_list:List[np.array],
                SVD_threshold:float = 0.01,
                print_rank:bool = True,
                SVD_max_rank:Optional[int] = None,
                order:int = 5,
                stream_transforms:List = ["time_enhance", "min_max_normalize"],
                n_jobs:int = 1,
                disable_tqdm:bool = False
                ):
        """Callable class which computes the anomaly distance with respect to a corpus of 
        variable length, d-dimensional time series, via truncated signature features.

        stream_list (List[np.array]): List of time series of shape (N_i, d).
                    SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
                    print_rank (bool): If true, prints out the SVD rank after thresholding.
                    SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
                    order (int): The order of the truncated signature features.
                    stream_transforms (List): A list of transformation names to be applied. Has to be 
                                    subset of ["time_enhance", "min_max_normalize", "lead_lag", "invisibility"].
                    n_jobs (int): Number of parallel jobs to run in the signature calculation.
                    disable_tqdm (bool): Whether to disable the progress bar.
        """
        sigs = streams_to_sigs(stream_list, order, stream_transforms, disable_tqdm, n_jobs=n_jobs)
        sigs = np.array(sigs)
        self.RnConfScore = RnConformanceScore(sigs, SVD_threshold, print_rank, SVD_max_rank)
        self.order = order
        self.stream_transforms = stream_transforms
    
    def __call__(self, 
                 stream:np.array,
                 method:str = "conformance") -> np.array:
        """ Returns the anomaly distance of a stream with respect to the corpus.
            The same stream transformations are applied to the input stream as to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            stream (np.ndarray): Feature vector of same dimension as the vectors in the corpus.
            method (str): Either "mahalanobis", "conformance", or "both".

        Returns:
            float: Anomaly distance.
        """
        stream = transform_stream(stream, self.stream_transforms)
        y = iisignature.sig(stream, self.order)
        return self.RnConfScore(y, method)


############################################################################################## |
############################### Kernelized Conformance Score  ################################ |
############################################################################################## \/


class KernelizedConformanceScore(BaseclassConformanceScore):
    def __init__(self, 
                corpus:List,
                kernel:Callable,
                SVD_threshold:float = 0.0,
                print_rank:bool = True,
                SVD_max_rank:Optional[int] = None,
                disable_Gram_tqdm:bool = False,
                n_jobs:int = 1,
                ):
        """Callable class which computes the kernelized anomaly score with respect to a corpus of data.

        Args:
            corpus (List): List of elements beloning to the same class.
            kernel (Callable): A kernel function which takes two elements and returns a non-negative scalar.
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            print_rank (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
            n_jobs (int): Number of parallel jobs to run in the kernel calculations.
            disable_Gram_tqdm (bool): Whether to disable the tqdm progress bar for the Gram calculations.
        """
        self.corpus = corpus
        self.kernel = kernel
        self.n_jobs = n_jobs

        #compute kernel Gram matrix
        inner_prod_Gram_matrix = pairwise_kernel_gram(corpus, corpus, kernel, 
                        sym=True, n_jobs=n_jobs, disable_tqdm=disable_Gram_tqdm)

        super().__init__(inner_prod_Gram_matrix, SVD_threshold, print_rank, SVD_max_rank)
    
    def __call__(self, 
                 new_sample:np.array,
                 method:str = "conformance",
                 disable_tqdm:bool = True) -> np.array:
        """ Returns the kernelized anomaly distance with respect to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            new_sample: Time series of shape (T, d).
            method (str): Either "mahalanobis", "conformance", or "both".
            disable_tqdm (bool): Whether to disable the tqdm progress bar.

        Returns:
            float: Anomaly distance.
        """
        # kernel as inner product
        inner_products = pairwise_kernel_gram([new_sample], self.corpus, self.kernel, 
                            sym=False, n_jobs=self.n_jobs, disable_tqdm=disable_tqdm)
        inner_products = np.array(inner_products)
        return self._anomaly_distance(inner_products, method)
    

############################################################################################## |
######### PDE Untruncated Signature Conformance Score for VARIABLE LENGTH timeseries ######### |
############################################################################################## \/


class SigKernelConformanceScore(KernelizedConformanceScore):
    def __init__(self, 
                stream_list:List[np.array],
                SVD_threshold:float = 0.01,
                print_rank:bool = True,
                SVD_max_rank:Optional[int] = None,
                static_kernel = sigkernel.LinearKernel(),
                dyadic_order_pde_solver:int = 3,
                stream_transforms:List = ["time_enhance", "min_max_normalize"],
                disable_Gram_tqdm:bool = False,
                n_jobs:int = 1,
                ):
        """Callable class which computes the full untruncated anomaly distance with respect to
           a corpus of variable length, d-dimensional time series, via the signature kernel.

        Args:
            stream_list (List[np.array]): List of time series of shape (N_i, d).
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            print_rank (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
            static_kernel: Base kernel for the signature kernel. Either sigkernel.RBFKernel(sigma), or sigkernel.LinearKernel().
            dyadic_order_pde_solver (int): Dyadic order for PDE solver (int > 0, higher = more accurate but slower)
            stream_transforms (List): A list of transformation names to be applied. Has to be 
                            subset of ["time_enhance", "min_max_normalize", "lead_lag", "invisibility"].
            disable_Gram_tqdm (bool): Whether to disable the tqdm progress bar for the Gram calculations.
            n_jobs (int): Number of parallel jobs to run in the kernel calculations.
        """
        sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order_pde_solver)
        kernel = lambda stream1, stream2 : sig_kernel.compute_kernel(
                                stream_to_torch(stream1, stream_transforms), 
                                stream_to_torch(stream2, stream_transforms)
                                ).numpy()[0]
        super().__init__(stream_list, kernel, SVD_threshold, print_rank, 
                            SVD_max_rank, disable_Gram_tqdm, n_jobs,)


############################################################################################## |
###### PDE Untruncated Signature Conformance Score for EQUAL LENGTH timeseries (faster?) ##### |
############################################################################################## \/


class EqualLengthSigKernelConformanceScore(BaseclassConformanceScore):
    def __init__(self, 
                stream_list:np.array,
                SVD_threshold:float = 0.01,
                print_rank:bool = True,
                SVD_max_rank:Optional[int] = None,
                static_kernel = sigkernel.LinearKernel(),
                dyadic_order_pde_solver:int = 3,
                stream_transforms:List = ["time_enhance", "min_max_normalize"],
                max_batch:int = 100,
                ):
        """Callable class which computes the full untruncated anomaly distance with respect to
           a corpus of equal length, d-dimensional time series, via the signature kernel.

        Args:
            stream_list (List[np.array]): List of time series of shape (N_i, d).
            SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
            print_rank (bool): If true, prints out the SVD rank after thresholding.
            SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
            static_kernel: Base kernel for the signature kernel. Either sigkernel.RBFKernel(sigma), or sigkernel.LinearKernel().
            dyadic_order_pde_solver (int): Dyadic order for PDE solver (int > 0, higher = more accurate but slower)
            stream_transforms (List): A list of transformation names to be applied. Has to be 
                            subset of ["time_enhance", "min_max_normalize", "lead_lag", "invisibility"].
            max_batch (int): Batch size in sig kernel computations.
        """
        #save transformed streams as torch tensor
        corpus_size, T, d = stream_list.shape
        self.streams = torch.from_numpy(np.array(
            [transform_stream(stream, stream_transforms) for stream in stream_list]
            )).detach()
        self.sigkernel_obj = sigkernel.SigKernel(static_kernel, dyadic_order_pde_solver)
        self.stream_transforms = stream_transforms
        self.max_batch = max_batch

        #compute kernel Gram matrix
        inner_prod_Gram_matrix = self.sigkernel_obj.compute_Gram(self.streams, self.streams, sym=True, max_batch=max_batch)
        inner_prod_Gram_matrix = inner_prod_Gram_matrix.numpy()

        super().__init__(inner_prod_Gram_matrix, SVD_threshold, print_rank, SVD_max_rank)
    

    def __call__(self, 
                 new_stream,
                 method:str = "conformance") -> np.array:
        """ Returns the kernelized anomaly distance with respect to the corpus.
            Uses either conformance score (nearest neighbour variance distance), 
            or Mahalanobis distance (variance distance to the mean).

        Args:
            new_sample: A new sample to compute the kernelized anomaly distance for.
            method (str): Either "mahalanobis", "conformance", or "both".

        Returns:
            float: Anomaly distance.
        """
        # kernel as inner product
        transformed_sample = stream_to_torch(new_stream, self.stream_transforms)
        inner_products = self.sigkernel_obj.compute_Gram(transformed_sample, self.streams, sym=False, max_batch=self.max_batch)
        inner_products = inner_products.squeeze().numpy()
        return self._anomaly_distance(inner_products, method)


if __name__ == "__main__":
    import time

    #Let corpus be a multivariate normal distribution.
    np.random.seed(99)
    N_corpus = 100
    d = 50
    M = 20
    mean = np.random.randn(d) / np.sqrt(d)
    cov = np.random.randn(M, d) / np.sqrt(d)
    cov = cov.T @ cov
    corpus = np.random.multivariate_normal(mean, cov, size=(N_corpus))
    in_sample = np.random.multivariate_normal(mean, cov)

    #Take a sample from a different distribution
    out_of_sample = np.random.multivariate_normal(mean-2, cov)

    #Test the anomaly distance
    def anomaly_test(name, corpus, in_sample, out_of_sample):
        print("{}:".format(name))
        start = time.time()
        if name == "Rn":
            confScorer = RnConformanceScore(corpus, SVD_threshold=0.01)
        elif name == "Truncated Signature":
            confScorer = TruncSigConformanceScore(corpus, order=5, stream_transforms=[], SVD_threshold=0.01)
        elif name == "Signature Kernel":
            confScorer = SigKernelConformanceScore(corpus, stream_transforms=[], SVD_threshold=0.01)
        elif name == "Signature Kernel (equal length, optimized)":
            confScorer = EqualLengthSigKernelConformanceScore(corpus, stream_transforms=[], SVD_threshold=0.01, max_batch=34)
        print("Anomaly distance for new sample, same distribution:     ", confScorer(in_sample))
        print("Anomaly distance for new sample, different distribution:", confScorer(out_of_sample))
        print("Time taken: {}\n\n".format(time.time()-start))

    anomaly_test("Rn", corpus, in_sample, out_of_sample)

    #Treat the corpus as a list of time series of shape (25, 2),
    #that is, time series of length 25, and dimension 2.
    corpus = corpus.reshape(N_corpus, 25, 2)
    in_sample = in_sample.reshape(25, 2)
    out_of_sample = out_of_sample.reshape(25, 2)

    anomaly_test("Truncated Signature", corpus, in_sample, out_of_sample)
    anomaly_test("Signature Kernel", corpus, in_sample, out_of_sample)
    anomaly_test("Signature Kernel (equal length, optimized)", corpus, in_sample, out_of_sample)