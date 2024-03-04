import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Literal


def z_score_normalize(train:np.ndarray, 
                      test:np.ndarray,
                      EPS:float = 0.0001):
    """Normalize train and test across axis=0 using mean and std
    of train set only."""
    mean = np.mean(train, axis=0, keepdims=True)
    std = np.std(train, axis=0, keepdims=True)
    train = (train - mean) / (std+EPS)
    test = (test - mean) / (std+EPS)
    return train, test


def check_ndim_more(X:np.ndarray, 
                    ndim:int):
    """Check if X has more than 'ndim' dimensions."""
    if X.ndim <= ndim:
        raise ValueError(f"X must have at least {ndim+1} dimensions. Current shape:", X.shape)


def avg_pool_time(X:np.ndarray,
                  pool_size:int = 1):
    """Average pools the time dimension of X by pool_size.
    
    Args:
        X (np.ndarray): Array of shape (..., T, d).
        pool_size (int): Size of the pool.
    
    Returns: 
        Array of shape (..., new_T, d).
    """
    check_ndim_more(X, 1)
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape

    new_T = T // pool_size
    reshaped = X[:, :new_T*pool_size, :].reshape(N, new_T, pool_size, d)
    pooled = np.mean(reshaped, axis=2)
    return pooled.reshape(original_shape[:-2] + (new_T, d))


def augment_time(X:np.ndarray,
                 min_val:float = 0.0,
                 max_val:float = 1.0):
    """Add time dimension to X with values between min_val and max_val.

    Args:
        X (np.ndarray): Array of shape (..., T, d).
        min_val (float): Minimum value of time.
        max_val (float): Maximum value of time.
    
    Returns: 
        Array of shape (..., T, d+1).
    """
    check_ndim_more(X, 1)
    original_shape = X.shape
    T, d = X.shape[-2:]
    X = X.reshape(-1, T, d)

    time = np.linspace(min_val, max_val, T)
    time = time.reshape(T, 1)
    time = np.repeat(time, X.shape[0], axis=-1) #Shape (T,N)
    time = time.T[:,:, None] #Shape (N, T, 1)
    X = np.concatenate([X, time], axis=-1)
    return X.reshape(original_shape[:-1] + (d+1,))


def add_basepoint_zero(X:np.ndarray):
    """Add basepoint zero to X in the time dimension.
    
    Args:
        X (np.ndarray): Array of shape (..., T, d).
        
    Returns: 
        Array of shape (..., T+1, d).
    """
    check_ndim_more(X, 1)
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    N, T, d = X.shape    

    basepoint = np.zeros((N, 1, d))
    X = np.concatenate([basepoint, X], axis=1)
    return X.reshape(original_shape[:-2] + (T+1, d))


def normalize_streams(train:np.ndarray, 
                      test:np.ndarray,
                      max_T:int = 70
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

    # clip to avoid numerical instability for poly and linear sigs
    c = 5.0
    train = np.clip(train, -c, c)
    test = np.clip(test, -c, c)

    # Add basepoint and augment time
    train = add_basepoint_zero(train)
    test = add_basepoint_zero(test)
    train = augment_time(train)
    test = augment_time(test)
    return train, test