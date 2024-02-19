import sklearn.preprocessing
import numpy as np
import iisignature
from tqdm import tqdm
from joblib import Memory, Parallel, delayed

from typing import List, Optional, Dict, Set, Callable


def transform_stream(raw_stream:np.ndarray, 
                     stream_transforms:List = ["time_enhance", "min_max_normalize", "lead_lag", "invisibility"],
                     time_normalization_factor:Optional[int] = None,):
    """Transforms the raw stream data using a series of specified transformations.

    Args:
        raw_stream (np.ndarray): The raw stream of shape (N, d) to be transformed.
        stream_transforms (List): A list of transformation names to be applied. Has to be 
                                  a subset of ["time_enhance", "min_max_normalize", "lead_lag", "invisibility"].
        time_normalization_factor (Optional[int]): Added time dimension will range from 0 to len(stream) / time_normalization_factor.
                                                       Recommended to be the maximum sequence length in the dataset, due to the signature
                                                       being an exponential. If None, normalize to [0,1].
    Returns:
        numpy.ndarray: The transformed stream data.
    """
    stream = raw_stream
    if "min_max_normalize" in stream_transforms:
        stream = sklearn.preprocessing.MinMaxScaler().fit_transform(stream)
    if "time_enhance" in stream_transforms:
        N,d=stream.shape
        factor = time_normalization_factor if time_normalization_factor else N
        time = np.linspace(0, N/factor, N)
        stream = np.column_stack((stream, time))
    if "lead_lag" in stream_transforms:
        stream = np.repeat(stream, 2, axis=0)
        stream = np.column_stack((stream[1:, :], stream[:-1, :]))
    if "invisibility" in stream_transforms:
        N,d=stream.shape
        stream = np.vstack(((stream, stream[-1], np.zeros_like(stream[-1]))))
        stream = np.column_stack((stream, np.append(np.ones(N-2), [0, 0])))
    return stream


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def prepare_RS_matrices(d:int, #dimension of path
                        K:int, #dimension of RS
                        seed:Optional[int]=None):
    if seed is not None:
        np.random.seed(seed)
    A = np.random.normal(0, 1/K, size=(d, K, K))
    b = np.random.normal(0, 1, size=(d, K))
    RS_0 = np.random.normal(0, 1, size=(K))
    return A, b, RS_0


def randomized_signature(stream:np.ndarray,
                         A:np.ndarray,
                         b:np.ndarray,
                         RS_0:np.array,
                        ) -> np.array:
    """returns the readout of S_t+1 = S_t + sum_i( A_i sigma(S_t + b_i)dX_t^i )

    Args:
        stream (np.ndarray): Stream of shape (N,d)
        A (np.ndarray): Array of shape (d,K,K)
        b (np.ndarray): Array of shape (d, K)
        RS_0 (np.ndarray): Initial value of RS, shape (K)
    """
    dX = np.diff(stream, axis=0)
    N,d = dX.shape
    RS = RS_0
    for dXt in dX:
        RS = RS + np.sum( [A[i] @ sigmoid(RS + b[i])*dXt[i] for i in range(d)] )
    return RS


def streams_to_randomized_sigs(stream_list:List[np.array],
                               K:int = 64, 
                               stream_transform:Callable[[np.array], np.array] = lambda stream : stream,
                               seed:Optional[int] = None,
                               enable_tqdm:bool = True,
                               **parallel_kwargs
                              ) -> List[np.array]:
    """Takes in a list of time series of shape (N_i, d) where N_i is allowed to vary,
       and transforms the time series into a feature vector by the use of randomized signatures.

    Args:
        stream_list (List[np.array]): List of time series of shape (N_i, d).
        K (int): Dimension of the randomized signature.
        stream_transform (Callable[[np.array], np.array]): Transformation applied to raw stream.
        seed (Optional[int]): Optional seed for the randomized signature algorithm.
        enable_tqdm (bool): If true, enables the tqdm progress bar.
        **parallel_kwargs: Params for joblib parallelisation.

    Returns:
        List[np.array]: List of feature vectors.
    """
    _, d = stream_list[0].shape
    A, b, RS_0 = prepare_RS_matrices(d, K, seed)
    func = lambda x : randomized_signature(stream_transform(x), A, b, RS_0)

    stream_list = Parallel(**parallel_kwargs)(
        delayed(func)(stream) 
        for stream in tqdm(stream_list, disable = not enable_tqdm, desc="Calculating randomized signatures"))
    
    return stream_list


def streams_to_sigs(stream_list:List[np.ndarray], 
                    order:int = 3, 
                    stream_transforms:List = [],
                    disable_tqdm:bool = False,
                    **parallel_kwargs
                   ) -> List[np.array]:
    """Takes in a list of time series of shape (N_i, d) where N_i is allowed to vary,
       and transforms the time series into feature vectors by the signature transform.

    Args:
        stream_list (List[np.ndarray]): List of time series of shape (N_i, d).
        order (int): Order of the signature transform.
        stream_transforms (List): A list of transformation names to be applied. Has to be 
                          subset of ["time_enhance", "min_max_normalize", "lead_lag", "invisibility"].
        disable_tqdm (bool): If true, disables the tqdm progress bar.
        **parallel_kwargs: Params for joblib parallelisation.

    Returns:
        np.ndarray: Array of truncated signatures.
    """
    stream_transform = lambda x : transform_stream(x, stream_transforms) 
    sig_fun = lambda x : iisignature.sig(stream_transform(x), order)

    sigs = Parallel(**parallel_kwargs)(
        delayed(sig_fun)(stream) 
        for stream in tqdm(stream_list, disable = disable_tqdm, desc="Calculating signatures"))
    
    return np.array(sigs)


# ###########################################
# ### Playing around with sigkernel
# ###########################################

# import torch
# import sigkernel

# # Specify the static kernel (for linear kernel use sigkernel.LinearKernel())
# static_kernel = sigkernel.RBFKernel(sigma=0.5)

# # Specify dyadic order for PDE solver (int > 0, default 0, the higher the more accurate but slower)
# dyadic_order = 3

# # Specify maximum batch size of computation; if memory is a concern try reducing max_batch, default=100
# max_batch = 11

# # Initialize the corresponding signature kernel
# signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

# # Synthetic data
# batch, len_x, len_y, dim = 5, 33, 33, 3
# torch.manual_seed(0)
# X = torch.rand((batch,len_x,dim), dtype=torch.float64, device='cpu') # shape (batch,len_x,dim)
# Y = torch.rand((batch,len_y,dim), dtype=torch.float64, device='cpu') # shape (batch,len_y,dim)
# Z = torch.rand((batch,len_x,dim), dtype=torch.float64, device='cpu') # shape (batch,len_y,dim)

# # Compute signature kernel "batch-wise" (i.e. k(x_1,y_1),...,k(x_batch, y_batch))
# K = signature_kernel.compute_kernel(X,Y,max_batch)

# # Compute signature kernel Gram matrix (i.e. k(x_i,y_j) for i,j=1,...,batch), also works for different batch_x != batch_y)
# G = signature_kernel.compute_Gram(X,X,sym=True, max_batch=max_batch)
# print(G)



##############################################################
################ Playing around with Ksig ####################
##############################################################

# import numpy as np
# import ksig

# # Number of signature levels to use.
# n_levels = 5 

# # Use the RBF kernel for vector-valued data as static (base) kernel.
# static_kernel = ksig.static.kernels.RBFKernel() 

# # Instantiate the signature kernel, which takes as input the static kernel.
# sig_kernel = ksig.kernels.SignatureKernel(n_levels, static_kernel=static_kernel)

# # Generate 10 sequences of length 50 with 5 channels.
# n_seq, l_seq, n_feat = 10, 50, 5 
# X = np.random.randn(n_seq, l_seq, n_feat)

# # Sequence kernels take as input an array of sequences of ndim == 3,
# # and work as a callable for computing the kernel matrix. 
# K_XX = sig_kernel(X)  # K_XX has shape (10, 10).

# # The diagonal kernel entries can also be computed.
# K_X = sig_kernel(X, diag=True)  # K_X has shape (10,).

# # Generate another array of 8 sequences of length 20 and 5 features.
# n_seq2, l_seq2 = 8, 20
# Y = np.random.randn(n_seq2, l_seq2, n_feat)

# # Compute the kernel matrix between arrays X and Y.
# K_XY = sig_kernel(X, Y)  # K_XY has shape (10, 8)