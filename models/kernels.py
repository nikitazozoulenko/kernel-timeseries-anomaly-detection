import numpy as np
from typing import List, Callable, Optional, Any
from numba import njit
import numba as nb
from joblib import Memory, Parallel, delayed
from tqdm import tqdm

from scipy.interpolate import interp1d


def pairwise_kernel_gram(X:List, 
                         Y:List, 
                         pairwise_kernel:Callable, 
                         sym:bool = False, 
                         n_jobs:int = 1, 
                         verbose:bool = False,
                         ) -> np.ndarray:
    """Calculates the kernel Gram matrix k(X_i, Y_j) of two collections X and Y
    using joblib.Parallel for parallelization.

    Args:
        X (List): List of elements
        Y (List): List of elements
        pairwise_kernel (Callable): Takes in two elements and outputs a value.
        sym (bool): If true, make Gram matrix symmetric.
        n_jobs (int): Number of parallel jobs to run.
        verbose (bool): Whether to enable the tqdm progress bar.
    """
    #Create indices to loop over
    N, M = len(X), len(Y)
    if sym:
        if X is not Y:
            raise ValueError("If sym=True, X and Y must be the same list.")
        indices = np.stack(np.triu_indices(N)).T #triangular pairs
    else:
        indices = np.stack(np.meshgrid(np.arange(N), np.arange(M))).T.reshape(-1,2)

    #Calculate kernel Gram matrix
    inner_products = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(pairwise_kernel)(X[i], Y[j]) 
        for i,j in tqdm(indices, disable = not verbose, desc="Kernel Gram Matrix"))

    #Populate matrix
    inner_prod_Gram_matrix = np.zeros((*inner_products[0].shape, N,M), 
                                      dtype=np.float64)
    for (i,j), val in zip(indices, inner_products):
        inner_prod_Gram_matrix[..., i,j] = val
        if sym:
            inner_prod_Gram_matrix[..., j,i] = val

    return inner_prod_Gram_matrix


##########################################################################
######################## Static Kernels on R^d ###########################
##########################################################################


def _check_gram_dims(X:np.ndarray, 
                     Y:np.ndarray,
                     diag:bool = False,):
    """Stacks the input into a Gram matrix shape (N1, N2, ..., d) or
    into a diagonal Gram shape (N1, ..., d) if diag and N1==N2.

    Args:
        X (np.ndarray): Shape (N1, ... , d).
        Y (np.ndarray): Shape (N2, ... , d).
        diag (bool): If True, use diagonal Gram shape.
    """
    len1 = len(X.shape)
    len2 = len(Y.shape)
    if (len1<2) or (len2<2):
        raise ValueError("X and Y must have at least 2 dimensions, found {len1} and {len2}.")
    if X.shape[1:] != Y.shape[1:]:
        raise ValueError("X and Y must have the same dimensions except for the first axis.")

    N1 = X.shape[0]
    N2 = Y.shape[0]
    if diag and N1!=N2:
        raise ValueError("If 'diag' is True, X and Y must have the same number of samples.")



def linear_kernel_gram(X:np.ndarray, 
                       Y:np.ndarray,
                       diag:bool = False,
                       divide_by_dims:bool = True,
                       custom_factor:Optional[float] = None,
                       ):
    """Computes the Rd inner product matrix <x_i, y_j> or diagonal <x_i, y_i>.
    The inputs dimensions can only differ in the first axis.
    
    Args:
        X (np.ndarray): Shape (N1, ... , d).
        Y (np.ndarray): Shape (N2, ... , d).
        diag (bool): If True, computes the diagonal of the gram matrix.
        divide_by_dims (bool): If True, divides the result by the dimension d.
        custom_factor (Optional[float]): If not None, ignores 'divide_by_dims' and 
                               multiplies the result by this factor instead.

    Returns:
        np.ndarray: Array of shape (N1, N2, ...) or (N1, ...) if diag=True.
    """
    _check_gram_dims(X, Y, diag)
    if diag:
        #out_i... = sum(X_i...k * Y_i...k)
        out = np.einsum('i...k,i...k -> i...', X, Y)
    else:
        #out_ij... = sum(X_i...k * Y_j...k)
        out = np.einsum('i...k,j...k -> ij...', X, Y)
    
    if custom_factor is not None:
        out = out * custom_factor
    elif divide_by_dims:
        d = X.shape[-1]
        out = out/d
    return out



def rbf_kernel_gram(X:np.ndarray, 
                    Y:np.ndarray,
                    sigma:float,
                    diag:bool = False,
                    divide_by_dims:bool = True,
                    custom_factor:Optional[float] = None,
                    ):
    """Computes the RBF gram matrix k(x_i, y_j) or diagonal k(x_i, y_i).
    The inputs dimensions can only differ in the first axis.
    
    Args:
        X (np.ndarray): Shape (N1, ... , d).
        Y (np.ndarray): Shape (N2, ... , d).
        sigma (float): RBF parameter
        diag (bool): If True, computes the diagonal of the gram matrix.
        divide_by_dims (bool): If True, normalizes the norm by the dimension d.
        custom_factor (Optional[float]): If not None, ignores 'divide_by_dims' and 
                               multiplies the result by this factor instead.

    Returns:
        np.ndarray: Array of shape (N1, N2, ...) or (N1, ...) if diag=True.
    """
    if diag:
        diff = X-Y
        norms_squared = linear_kernel_gram(diff, diff, diag=True, 
                                           divide_by_dims=divide_by_dims,
                                           custom_factor=custom_factor)
    else:
        xx = linear_kernel_gram(X, X, diag=True, divide_by_dims=divide_by_dims, custom_factor=custom_factor) #shape (N1, ...)
        xy = linear_kernel_gram(X, Y, diag=False, divide_by_dims=divide_by_dims, custom_factor=custom_factor) #shape (N1, N2, ...)
        yy = linear_kernel_gram(Y, Y, diag=True, divide_by_dims=divide_by_dims, custom_factor=custom_factor) #shape (N2, ...)
        norms_squared = -2*xy + xx[:, np.newaxis] + yy[np.newaxis, :] 

    return np.exp(-sigma * norms_squared)



def poly_kernel_gram(X:np.ndarray, 
                     Y:np.ndarray,
                     p:float, #eg 2 or 3
                     diag:bool = False,
                     divide_by_dims:bool = True,
                     custom_factor:Optional[float] = None,):
    """Computes the polynomial kernel (<x_i, y_j> + 1)^p.
    The inputs dimensions can only differ in the first axis.
    
    Args:
        X (np.ndarray): Shape (N1, ... , d).
        Y (np.ndarray): Shape (N2, ... , d).
        p (float): Polynomial degree.
        diag (bool): If True, computes the diagonal of the gram matrix.
        divide_by_dims (bool): If True, normalizes the norm by the dimension d.
        custom_factor (Optional[float]): If not None, ignores 'divide_by_dims' and
                                 multiplies the result by this factor instead.

    Returns:
        np.ndarray: Array of shape (N1, N2, ...) or (N1, ...) if diag=True.
    """
    xy = linear_kernel_gram(X, Y, diag, divide_by_dims, custom_factor)
    return (xy + 1)**p


#######################################################################################
################### time series Integral Kernel of static kernel ######################
#######################################################################################


def integral_kernel(s1: np.ndarray,
                    s2: np.ndarray,
                    static_diag_kernel:Callable,
                    )-> float:
    """Computes the integral kernel K(x, y) = \int_[0,1] k(x_t, y_t) dt 
    given static kernel and two piecewise linear paths.

    Args:
        s1 (np.ndarray): A time series of shape (T1, d).
        s2 (np.ndarray): A time series of shape (T2, d).
        static_diag_kernel_gram (Callable): Takes in two arrays of shape (M, d) 
                        and outputs the diagonal Gram <x_m, y_m> of shape (M).
    """
    #Find all breakpoints of the piecewise linear paths
    T1, d = s1.shape
    T2, d = s2.shape
    times = np.concatenate([np.linspace(0, 1, T1), np.linspace(0, 1, T2)])
    times = sorted(np.unique(times))

    #Add the extra breakpoints to the paths
    f1 = interp1d(np.linspace(0, 1, T1), s1, axis=0, assume_sorted=True)
    f2 = interp1d(np.linspace(0, 1, T2), s2, axis=0, assume_sorted=True)
    x = f1(times) #shape (len(times), d)
    y = f2(times)

    #calculate k(x_t, y_t) for each t
    Kt = static_diag_kernel(x, y)

    #return integral of k(x_t, y_t) dt
    return np.trapz(Kt, times)



def integral_kernel_gram(
        X:List[np.ndarray],
        Y:List[np.ndarray],
        static_kernel_gram:Callable, #either linear_kernel_gram or rbf_kernel_gram with "diag" argument
        fixed_length:bool,
        sym:bool = False,
        n_jobs:int = 1,
        verbose:bool = False,
    ):
    """Computes the Gram matrix K(X_i, Y_j) of the integral kernel 
    K(x, y) = \int k(x_t, y_t) dt.

    Args:
        static_kernel_gram (Callable): Gram kernel function taking in two ndarrays and
                    one boolean "diag" argument, see e.g. 'linear_kernel_gram' or 
                    'rbf_kernel_gram'.
        X (List[np.ndarray]): List of time series of shape (T_i, d).
        Y (List[np.ndarray]): List of time series of shape (T_j, d).
        fixed_length (bool): If True, uses the optimized kernels for fixed 
                                length time series.
        sym (bool): If True, computes the symmetric Gram matrix.
        n_jobs (int): Number of parallel jobs to run.
        verbose (bool): Whether to enable the tqdm progress bar.
    """
    if fixed_length:
        X = np.array(X)
        Y = np.array(Y)
        ijKt = static_kernel_gram(X, Y, False) #diag=False

        #return integral of k(x_t, y_t) dt for each pair x and y
        N1, T, d = X.shape
        return np.trapz(ijKt, dx=1/(T-1), axis=-1)
    else:
        static_ker = lambda a,b : static_kernel_gram(a,b, True) #diag=True
        pairwise_int_ker = lambda s1, s2 : integral_kernel(s1, s2, static_ker)
        return pairwise_kernel_gram(X, Y, pairwise_int_ker, sym, n_jobs, verbose)


############################################################################
################# signature kernels of static kernels ######################
############################################################################


def sig_kernel(s1:np.ndarray, 
               s2:np.ndarray, 
               order:int,
               static_kernel_gram:Callable = linear_kernel_gram,
               only_last:bool = True):
    """s1 and s2 are time series of shape (T_i, d)"""
    K = static_kernel_gram(s1, s2)
    nabla = K[1:, 1:] + K[:-1, :-1] - K[1:, :-1] - K[:-1, 1:]
    sig_kers = jitted_trunc_sig_kernel(nabla, order)
    if only_last:
        return sig_kers[-1]
    else:
        return sig_kers



@njit((nb.float64[:, ::1], nb.int64), fastmath=True, cache=True)
def reverse_cumsum(arr:np.ndarray, axis:int): #ndim=2
    """JITed reverse cumulative sum along the specified axis.
    (np.cumsum with axis is not natively supported by Numba)"""
    A = arr.copy()
    if axis==0:
        for i in np.arange(A.shape[0]-2, -1, -1):
            A[i, :] += A[i+1, :]
    else: #axis==1
        for i in np.arange(A.shape[1]-2, -1, -1):
            A[:,i] += A[:,i+1]
    return A



@njit((nb.float64[:, ::1], nb.int64), fastmath=True, cache=True)
def jitted_trunc_sig_kernel(nabla, order):
    """Given difference matrix nabla_ij = K[i+1, j+1] + K[i, j] - K[i+1, j] - K[i, j+1],
    computes the truncated signature kernel of all orders up to 'order'."""
    B = np.ones((order+1, order+1, order+1, *nabla.shape))
    for d in np.arange(order):
        for n in np.arange(order-d):
            for m in np.arange(order-d):
                B[d+1,n,m] = 1 + nabla/(n+1)/(m+1)*B[d, n+1, m+1]
                r1 = reverse_cumsum(nabla * B[d, n+1, 1] / (n+1), axis=0)
                B[d+1,n,m, :-1, :] += r1[1:, :]
                r2 = reverse_cumsum(nabla * B[d, 1, m+1] / (m+1), axis=1)
                B[d+1,n,m, :, :-1] += r2[:, 1:]
                rr = reverse_cumsum(nabla * B[d, 1, 1], axis=0)
                rr = reverse_cumsum(rr, axis=1)
                B[d+1,n,m, :-1, :-1] += rr[1:, 1:]

    #copy, otherwise all memory accumulates in for loop
    return B[1:,0,0,0,0].copy() 



def sig_kernel_gram(
        X:List[np.ndarray],
        Y:List[np.ndarray],
        order:int,
        static_kernel_gram:Callable,
        only_last:bool = True,
        sym:bool = False,
        n_jobs:int = 1,
        verbose:bool = False,
    ):
    """Computes the Gram matrix k_sig(X_i, Y_j) of the signature kernel,
    given the static kernel k(x, y) and the truncation order.

    Args:
        X (List[np.ndarray]): List of time series of shape (T_i, d).
        Y (List[np.ndarray]): List of time series of shape (T_j, d).
        static_kernel_gram (Callable): Gram kernel function taking in two ndarrays,
                            see e.g. 'linear_kernel_gram' or 'rbf_kernel_gram'.
        order (int): Truncation level of the signature kernel.
        only_last (bool): If False, returns results of all truncation levels up to 'order'.
        sym (bool): If True, computes the symmetric Gram matrix.
        n_jobs (int): Number of parallel jobs to run.
        verbose (bool): Whether to enable the tqdm progress bar.
    """
    pairwise_ker = lambda s1, s2 : sig_kernel(s1, s2, order, static_kernel_gram, only_last)
    return pairwise_kernel_gram(X, Y, pairwise_ker, sym, n_jobs, verbose)


