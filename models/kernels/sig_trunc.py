from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import TimeSeriesKernel, StaticKernel
from kernels.static_kernels import RBFKernel


@torch.jit.script
def cumsum_shift1(X:Tensor, dim:int):
    """
    Computes the cumulative sum of a tensor X, then shifts it by one
    and pads with a zero.
    
    Args:
        X (Tensor): Tensor of shape (..., T1, T2).
        dim (int): Dimension to cumsum over and shift.
    """
    Q = X.clone()
    if dim==-2:
        Q[..., 1:, :] = Q[..., :-1, :].cumsum(dim=-2)
        Q[..., 0, :] = 0
    elif dim==-1:
        Q[..., 1:] = Q[..., :-1].cumsum(dim=-1)
        Q[..., 0] = 0
    return Q


@torch.jit.script
def trunc_sigker_geoGEQ2(
        nabla:Tensor, 
        trunc_level:int, 
        geo_order:int,
        only_last:bool,
    ):
    """
    Computes the truncated signature kernel given a matrix 
    nabla[s,t] = K[s+1, t+1] + K[s, t] - K[s+1, t] - K[s, t+1].
    See Algo 6 in https://jmlr.org/papers/v20/16-314.html.

    Args:
        nabla (Tensor): Matrix of shape (..., T1, T2).
        trunc_level (int): Truncation level of the signature.
        geo_order (int): Geometric order of the rough path lift.
    """
    sh = nabla.shape
    A = torch.zeros(sh[:-2]+(geo_order, geo_order)+sh[-2:],
                    device=nabla.device, dtype=nabla.dtype)
    results = torch.empty( sh[:-2]+(trunc_level,),
                          device=nabla.device, dtype=nabla.dtype)
    for n in range(trunc_level):
        AA = A.clone()
        Asum0 = AA.sum(dim=-4)
        Asum1 = AA.sum(dim=-3)
        Asum01 = Asum0.sum(dim=-3)
        A[..., 0, 0, :, :] = nabla * (1+cumsum_shift1(cumsum_shift1(Asum01, dim=-1), dim=-2))
        
        d = min(n+1, geo_order)
        for r in range(1, d):
            A[..., r, 0, :, :] = 1/(r+1) * nabla * cumsum_shift1(Asum1[..., r-1, :, :], dim=-2)
            A[..., 0, r, :, :] = 1/(r+1) * nabla * cumsum_shift1(Asum0[..., r-1, :, :], dim=-1)

            for s in range(1, d):
                A[..., r, s, :, :] = 1/(r+1)/(s+1) * nabla * AA[..., r-1, s-1, :, :]
        # save
        results[..., n] = 1 + A.sum(dim = (-4, -3, -2, -1))
    
    if only_last:
        return results[..., -1]
    else:
        return results
    

@torch.jit.script
def trunc_sigker_geo1(
        nabla:Tensor, 
        trunc_level:int, 
        only_last:bool,
    ):
    """
    Computes the non-geometric truncated signature kernel given a matrix 
    nabla[s,t] = K[s+1, t+1] + K[s, t] - K[s+1, t] - K[s, t+1]. See Algo 
    3 in https://jmlr.org/papers/v20/16-314.html.

    Args:
        nabla (Tensor): Matrix of shape (..., T1, T2).
        trunc_level (int): Truncation level of the signature.
    """
    sh = nabla.shape
    A = nabla.clone()
    results = torch.empty(sh[:-2]+(trunc_level,),
                          device=nabla.device, dtype=nabla.dtype)
    results[..., 0] = 1 + A.sum(dim = (-2, -1))
    for n in range(1, trunc_level):
        A = nabla * cumsum_shift1(cumsum_shift1(A, dim=-2), dim=-1)
        results[..., n] = 1 + A.sum(dim = (-2, -1))

    if only_last:
        return results[..., -1]
    else:
        return results


class TruncSigKernel(TimeSeriesKernel):
    def __init__(
            self,
            static_kernel:StaticKernel = RBFKernel(),
            trunc_level:int = 5,
            geo_order:int = 1,
            only_last:bool = True,
            max_batch:int = 100,
            normalize:bool = False,
        ):
        """
        The truncated signature kernel of two time series of 
        shape (T_i, d) with respect to a static kernel on R^d.
        See https://jmlr.org/papers/v20/16-314.html. The parameter
        'geo_order' is the geometric order of the rough path lift, 
        where geo_order=trunc_level corresponds to the exact signature
        of the piecewise linear path, and geo_order=1 corresponds to
        a discretization of the signature. O(T^2(d + trunc_level*geo_order^2)) 
        time for each pair of time series.

        Args:
            static_kernel (StaticKernel): Static kernel on R^d.
            trunc_level (int): Truncation level of the signature kernel.
            geo_order (int): Geometric order of the rough path lift. Has
                to be less than or equal to 'trunc_level'.
            only_last (bool): If False, returns results of all truncation 
                levels up to 'trunc_level'.
            max_batch (int, optional): Max batch size for computations.
            normalize (bool, optional): If True, normalizes the kernel.
        """
        super().__init__(max_batch, normalize)
        assert geo_order <= trunc_level, "geo_order has to be less than or equal to trunc_level."
        assert geo_order > 0, "geo_order has to be greater than 0."
        self.static_kernel = static_kernel
        self.trunc_level = trunc_level
        self.geo_order = geo_order
        self.only_last = only_last


    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool,
        ):
        # nabla_st = K[s+1, t+1] + K[s, t] - K[s+1, t] - K[s, t+1] in time
        K = self.static_kernel.time_gram(X, Y, diag)
        nabla = K.diff(dim=-1).diff(dim=-2) # shape (N, T1, T2)
        if self.geo_order >= 2:
            return trunc_sigker_geoGEQ2(nabla, self.trunc_level, self.geo_order, self.only_last)
        else:
            return trunc_sigker_geo1(nabla, self.trunc_level, self.only_last)