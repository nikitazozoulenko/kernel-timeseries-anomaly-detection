from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor
import sigkernel
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import StaticKernel, TimeSeriesKernel
from kernels.static_kernels import RBFKernel


class CrisStaticWrapper:
    def __init__(
            self, 
            kernel: StaticKernel,
        ):
        """Wrapper for static kernels for Cris Salvi's sigkernel library"""
        self.kernel = kernel


    def batch_kernel(
            self, 
            X:Tensor, 
            Y:Tensor
        ) -> Tensor:
        """
        Outputs k(X^i_t, Y^j_t)

        Args:
            X (Tensor): Tensor of shape (N, T1, d)
            Y (Tensor): Tensor of shape (N, T2, d)

        Returns:
            Tensor: Tensor of shape (N, T1, T2)
        """
        return self.kernel.time_gram(X, Y, diag=True)


    def Gram_matrix(
            self, 
            X: Tensor, 
            Y: Tensor
        ) -> Tensor:
        """
        Outputs k(X^i_s, Y^j_t)
        
        Args:
            X (Tensor): Tensor of shape (N1, T1, d)
            Y (Tensor): Tensor of shape (N2, T2, d)
        
        Returns:
            Tensor: Tensor of shape (N1, N2, T1, T2)
        """
        return self.kernel.time_gram(X, Y, diag=False)
    
    

class SigPDEKernel(TimeSeriesKernel):
    def __init__(
            self,
            static_kernel: StaticKernel = RBFKernel(),
            dyadic_order:int = 1,
            max_batch:int = 100,
            normalize: bool = False,
        ):
        """
        Signature PDE kernel for timeseries (x_1, ..., x_T) in R^d,
        kernelized with a static kernel k : R^d x R^d -> R.

        Args:
            static_kernel (StaticKernel): Static kernel on R^d.
            dyadic_order (int, optional): Dyadic order in PDE solver. Defaults to 1.
            max_batch (int, optional): Max batch size for computations. Defaults to 10.
            normalize (bool, optional): If True, normalizes the kernel. Defaults to False.
        """
        super().__init__(max_batch, normalize)
        self.static_wrapper = CrisStaticWrapper(static_kernel)
        self.dyadic_order = dyadic_order
        self.sig_ker = sigkernel.SigKernel(self.static_wrapper, dyadic_order)


    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool,
        ):
        # sigker required float64, otherwise we get an error
        X = X.double()
        Y = Y.double()
        
        #get gram matrix
        if diag:
            gram = self.sig_ker.compute_kernel(X, Y)
        else:
            gram = self.sig_ker.compute_Gram(X, Y)
        
        #recast and return
        return gram.to(dtype=X.dtype)

