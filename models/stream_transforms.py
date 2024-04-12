import torch
from torch import Tensor
from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable



def assert_ndim_geq(X:Tensor, ndim:int):
    """Check if X has more than 'ndim' dimensions."""
    if X.ndim < ndim:
        raise ValueError(f"X must have at least {ndim+1} dimensions. Current shape:", X.shape)
    return True



def z_score_normalize(
        train:Tensor,
        test:Tensor,
        epsilon:float = 0.0001,
    ):
    """
    Normalize 'train' and 'test' across axis=0 using mean and std
    of 'train' only.

    Args:
        train (Tensor): Tensor with shape (N, ...).
        test (Tensor): Tensor with shape (N, ...).
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        'train' and 'test' normalized by the mean and std of 'train'.
    """
    assert_ndim_geq(train, 1)
    mean = torch.mean(train, axis=0, keepdims=True)
    std = torch.std(train, axis=0, keepdims=True)
    train = (train - mean) / (std+epsilon)
    test = (test - mean) / (std+epsilon)
    return train, test



def avg_pool_time(
        X:Tensor,
        pool_size:int = 1,
    ):
    """
    Average pools the time dimension of X by pool_size.
    
    Args:
        X (Tensor): Tensor with shape (..., T, d).
        pool_size (int): Size of the pool.
    
    Returns: 
        Tensor with shape (..., new_T, d).
    """
    # reshape to 3D
    assert_ndim_geq(X, 2)
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape

    # pool time dimension
    new_T = T // pool_size
    X_grouped = X[:, :new_T*pool_size, :].reshape(N, new_T, pool_size, d)
    pooled = X_grouped.mean(axis=2)

    # reshape back to original shape
    return pooled.reshape(original_shape[:-2] + (new_T, d))



def augment_time(
        X:Tensor,
        min_val:float = 0.0,
        max_val:float = 1.0,
    ):
    """
    Add time channel/dim to 'X' with values uniformly between 
    'min_val' and 'max_val'.

    Args:
        X (Tensor): Tensor with shape (..., T, d).
        min_val (float): Minimum value of time.
        max_val (float): Maximum value of time.
    
    Returns: 
        Tensor with shape (..., T, d+1).
    """
    # reshape to 3D
    assert_ndim_geq(X, 2)
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape
    dtype = X.dtype
    device = X.device

    # concat time dimension. NOTE torch.repeat works like np.tile
    time = torch.linspace(min_val, max_val, T, dtype=dtype, device=device)
    time = time.repeat(N, 1)[:,:, None] #shape (N, T, 1)
    X = torch.concatenate([X, time], axis=-1)

    # reshape back to original shape
    return X.reshape(original_shape[:-1] + (d+1,))



def add_basepoint_zero(
        X:Tensor,
        first:bool = True,
    ):
    """
    Add basepoint zero to 'X' in the time dimension.
    
    Args:
        X (Tensor): Tensor with shape (..., T, d).
        first (bool): If True, add basepoint at the beginning of time.
                      If False, add basepoint at the end of time.
        
    Returns: 
        Tensor with shape (..., T+1, d).
    """
    # reshape to 3D
    assert_ndim_geq(X, 2)
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape
    dtype = X.dtype
    device = X.device

    # add basepoint
    basepoint = torch.zeros((N, 1, d), dtype=dtype, device=device)
    v = [basepoint, X] if first else [X, basepoint]
    X = torch.concatenate(v, axis=1)

    # reshape back to original shape
    return X.reshape(original_shape[:-2] + (T+1, d))



def I_visibility_transform(X:Tensor):
    """
    Performs the I-visiblity transform on 'X', see page 5 of
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9412642

    Args:
        X (Tensor): Tensor with shape (..., T, d).
        
    Returns: 
        Tensor with shape (..., T+2, d+1).
    """
    # reshape to 3D
    assert_ndim_geq(X, 2)
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape
    dtype = X.dtype
    device = X.device

    # (vec(0), 0) (x_1, 0) then (x_1, 1) (x_2, 1) ...
    X = add_basepoint_zero(X, first=True) # start of time
    start = torch.concatenate([X[:, 0:2, :], 
                               torch.zeros((N, 2, 1), dtype=dtype, device=device)], 
                            axis=-1)
    rest = torch.concatenate([X[:, 1:, :], 
                              torch.ones((N, T, 1), dtype=dtype, device=device)], 
                            axis=-1)
    X = torch.concatenate([start, rest], axis=1)

    # reshape back to original shape
    return X.reshape(original_shape[:-2] + (T+2, d+1))



def T_visibility_transform(X:Tensor):
    """
    Performs the T-visiblity transform on 'X', see page 5 of
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9412642

    Args:
        X (Tensor): Tensor with shape (..., T, d).
        
    Returns: 
        Tensor with shape (..., T+2, d+1).
    """
    # reshape to 3D
    assert_ndim_geq(X, 2)
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape
    dtype = X.dtype
    device = X.device

    # (x_1, 1) (x_2, 1) ... then (x_T, 0) (vec(0), 0)
    X = add_basepoint_zero(X, first=False) # end of time
    rest = torch.concatenate([X[:, :-1, :], 
                              torch.ones((N, T, 1), dtype=dtype, device=device)], 
                            axis=-1)
    end = torch.concatenate([X[:, -2:, :], 
                             torch.zeros((N, 2, 1), dtype=dtype, device=device)],
                             axis=-1)
    X = torch.concatenate([rest, end], axis=1)

    # reshape back to original shape
    return X.reshape(original_shape[:-2] + (T+2, d+1))



def normalize_streams(train:Tensor, 
                      test:Tensor,
                      max_T:int = 100,
                      ):
    """Inputs are 3D arrays of shape (N, T, d) where N is the number of time series, 
    T is the length of each time series, and d is the dimension of each time series.
    Performs average pooling to reduce the length of the time series to at most max_T,
    z-score normalization, basepoint addition, and time augmentation.
    """
    # Make time series length smaller
    _, T, d = train.shape
    if T > max_T:
        pool_size = 1 + (T-1) // max_T
        train = avg_pool_time(train, pool_size)
        test = avg_pool_time(test, pool_size)

    # Normalize data by training set mean and std
    train, test = z_score_normalize(train, test)

    # clip to avoid numerical instability
    c = 5.0
    train = torch.clip(train, -c, c)
    test = torch.clip(test, -c, c)
    return train, test