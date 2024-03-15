from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
from torch import Tensor
import torch

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import TimeSeriesKernel, StaticKernel
from kernels.static_kernels import LinearKernel


class ReservoirKernel(TimeSeriesKernel):
    def __init__(
            self,
            tau:float,
            gamma:float,
            max_batch:int = 1000,
            normalize:bool = False,
        ):
        """
        The reservoir kernel of two time series of shape (T, d), 
        see https://arxiv.org/pdf/2212.14641.pdf. Time complexity
        O(N*N*T*d), where N is the batch size.

        Args:
            tau (float): |1/tau| is uniform bound for values of 
                all input time series.
            gamma (float): Kernel parameter.
        """
        super().__init__(max_batch, normalize)
        self.tau = tau
        self.gamma = gamma
        self.max_batch = max_batch
        self.lin_ker = LinearKernel()


    def _gram(
            self, 
            X: Tensor,
            Y: Tensor,
            diag: bool,
        ):
        # shape (N, N2, T) or (N, T) if diag=True
        state_space_gram = self.lin_ker(X, Y, diag) 
        prod = self.gamma**2 / (1 - self.tau**2 * state_space_gram)
        
        # A(t+1) = 1 + gamma^2 / (1 - tau^2 <x(t), y(t)>) * A(t)
        gram = prod.cumprod(dim=-1).sum(dim=-1)
        CLIP = 1e+5
        return torch.clip(1 + gram, -CLIP, CLIP)
    