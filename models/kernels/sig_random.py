from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
from torch.nn.functional import relu
from torch.nn.functional import tanh
import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import TimeSeriesKernel, StaticKernel
from kernels.static_kernels import RBFKernel


@torch.jit.script
def randomized_sig_tanh(
        X:Tensor,
        A:Tensor,
        b:Tensor,
        Y_0:Tensor,
    ):
    """
    Randomized signature of a (batched) time series X, with tanh
    activation function.

    Args:
        X (Tensor): Input tensor of shape (N, T, d).
        A (Tensor): Tensor of shape (M, M, d). Random matrix.
        b (Tensor): Tensor of shape (M, d). Random bias.
        Y_0 (Tensor): Initial value of the randomized signature.
            Tensor of shape (M).
    """
    N, T, d = X.shape
    diff = X.diff(dim=1) # shape (N, T-1, d)
    Y_0 = torch.tile(Y_0, (N, 1)) # shape (N, M)

    #iterate y[t+1] = y[t] + ...
    Z = torch.tensordot(tanh(Y_0), A, dims=1) + b[None] # shape (N, M, d)
    Y = Y_0 + (Z * diff[:, 0:1, :]).sum(dim=-1) # shape (N, M)
    for t in range(1, T-1):
        Z = torch.tensordot(tanh(Y), A, dims=1) + b[None]
        Y = Y + (Z * diff[:, t:t+1, :]).sum(dim=-1)
    return Y



class RandomizedSigKernel(TimeSeriesKernel):
    def __init__(
            self,
            n_features = 100,
            seed:int = 0,
            max_batch:int = 10000,
            normalize:bool = False,
        ):
        """
        The randomized signature kernel of two time series of 
        shape (T_i, d).

        Args:
            n_features (int): Number of features.
            seed (int): Random seed.
            max_batch (int, optional): Max batch size for computations.
            normalize (bool, optional): If True, normalizes the kernel.
        """
        super().__init__(max_batch, normalize)
        self.n_features = n_features
        self.seed = seed
        self.has_initialized = False


    def _init_given_input(
            self, 
            X: Tensor
        ):
        """
        Initializes the random matrices and biases used in the 
        randomized signature kernel.

        Args:
            X (Tensor): Example input tensor of shape (N, T, d) of 
                timeseries.
        """
        # Get shape, dtype and device info.
        N, T, d = X.shape
        device = X.device
        dtype = X.dtype
        
        # Create a generator and set the seed
        gen = torch.Generator(device=device).manual_seed(self.seed)
        
        # Initialize the random matrices and biases
        self.A = torch.randn(self.n_features, 
                             self.n_features, 
                             d, 
                             device=device,
                             dtype=dtype,
                             generator=gen
                             )
        self.b = torch.randn(self.n_features,
                             d,
                             device=device,
                             dtype=dtype,
                             generator=gen
                             )

        self.Y_0 = torch.randn(self.n_features,
                               device=device,
                               dtype=dtype,
                               generator=gen)
        self.has_initialized = True
        

    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool,
        ):
        if not self.has_initialized:
            self._init_given_input(X)
        
        feat_X = randomized_sig_tanh(X, self.A, self.b, self.Y_0)
        feat_Y = randomized_sig_tanh(Y, self.A, self.b, self.Y_0)

        if diag:
            return (feat_X * feat_Y).mean(dim=-1)
        else:
            return feat_X @ feat_Y.t() / self.n_features