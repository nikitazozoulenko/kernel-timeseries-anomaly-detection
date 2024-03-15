from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import TimeSeriesKernel, StaticKernel
from kernels.static_kernels import RBFKernel, LinearKernel
import random


def sigma_gak(
        X:Tensor,
        N_samples:int = 10000,
        seed:Optional[int] = None,
    ):
    """
    Computes the recommended sigma parameter for the GAK kernel,
    i.e. the med(|X^i_s, X^j_t|) * sqrt(T) for a dataset X.

    Args:
        X (Tensor): Tensor of shape (N, T, d).
        N_samples (int): Number of samples to use for the estimation.
        seed (int, optional): Seed for the random number generator. Defaults to None.
    """
    if seed is not None:
        torch.manual_seed(seed)
    N, T, d = X.shape
    N_samples = min(N_samples, N*T)
    indices = random.sample(range(N*T), N_samples)
    X = X.view(-1, d)[indices]

    lin_ker = LinearKernel()
    dists = lin_ker.squared_dist(X, X)
    return torch.sqrt(dists.median()) * T**0.5



@torch.jit.script
def gak_update_antidiag(
        logK:Tensor,
        s:int,
        t:int,
    ):
    """
    Function to be used in the computation of the GAK kernel.
    Note that s,t>0
    """
    logM00 = logK[..., s-1, t-1]
    logM01 = logK[..., s-1, t  ]
    logM10 = logK[..., s  , t-1]
    logMmax = torch.maximum(torch.maximum(logM00, logM01), logM10)
    logK[..., s, t] += logMmax + torch.log(
        torch.exp(logM00 - logMmax) +
        torch.exp(logM01 - logMmax) +
        torch.exp(logM10 - logMmax)
        )



#@torch.jit.script
def log_global_align(
        K:Tensor, 
    ):
    """
    See fig 2 in 
    https://icml.cc/2011/papers/489_icmlpaper.pdf

    Args:
        K (Tensor): Tensor of shape (..., T1, T2) of Gaussian
            kernel evaluations K(x_s, x_t).
        triangle_param (int): Parameter in the TGAK kernel.
    """
    # make infinitely divisible
    T1, T2 = K.shape[-2:]
    K = K / (2 - K)
    EPS = 1e-10
    logK = torch.log(torch.clamp(K, min=EPS))

    #first do s=0, t=0
    logK[..., :, 0] = logK[..., :, 0].cumsum(dim=-1)
    logK[..., 0, :] = logK[..., 0, :].cumsum(dim=-1)
    #iterate over antidiagonals with s,t>0
    for diag in range(2, T1+T2-1):
        futures : List[torch.jit.Future[None]] = []
        for s in range(max(1, diag - T2 + 1), min(diag, T1)):
            t = diag - s
            futures.append( torch.jit.fork(gak_update_antidiag, logK, s, t) )
        [torch.jit.wait(fut) for fut in futures]
    return logK[..., -1, -1]



########################################################  |
################### GAK Kernel class ###################  |
######################################################## \|/


class GlobalAlignmentKernel(TimeSeriesKernel):
    def __init__(
            self,
            static_kernel:StaticKernel = RBFKernel(),
            max_batch:int = 1000,
            normalize:bool = True,
        ):
        """
        The global alignment kernel of two time series of shape T_i, d), 
        with the respect to a static kernel on R^d. For details see
        https://icml.cc/2011/papers/489_icmlpaper.pdf. Time O(d*T^2) 
        for each pair of time series. Only stable for certain classes
        of static kernels, such as RBF. Note that the static kernel is 
        made into a 'infinitely divisible' kernel through K/(2-K).

        Args:
            static_kernel (StaticKernel): Static kernel on R^d.
        """
        super().__init__(max_batch, normalize)
        self.static_kernel = static_kernel
    

    @property
    def log_space(self):
        return True


    def _gram(
            self, 
            X: Tensor,
            Y: Tensor,
            diag: bool,
        ):
        # K shape (N, T1, T2)
        K = self.static_kernel.time_gram(X, Y, diag)
        return log_global_align(K)