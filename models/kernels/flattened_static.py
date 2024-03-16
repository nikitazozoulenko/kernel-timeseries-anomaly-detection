from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import StaticKernel, TimeSeriesKernel
from kernels.static_kernels import RBFKernel

#######################################################################################
################### Time series Integral Kernel of static kernel ######################
#######################################################################################


class FlattenedStaticKernel(TimeSeriesKernel):
    def __init__(
            self,
            static_kernel:StaticKernel = RBFKernel(),
            max_batch:int = 10000,
            normalize:bool = False,
        ):
        """
        Treats a time series as a big vector in R^(Td), where T is the 
        length of the time series and d is the state-space dimension.
        K(x,y) = k(flat(x), flat(y)).

        Args:
            static_kernel (StaticKernel): Static kernel on R^d.
        """
        super().__init__(max_batch, normalize)
        self.static_kernel = static_kernel


    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool,
        ):
        N, T, d = X.shape
        N2, T, d = Y.shape
        X_flat = X.reshape(N, -1)
        Y_flat = Y.reshape(N2, -1)
        return self.static_kernel(X_flat, Y_flat, diag)