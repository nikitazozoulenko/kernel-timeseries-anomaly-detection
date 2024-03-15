from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import itertools

import numpy as np
import torch
from joblib import Parallel, delayed
from torch import Tensor


def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target
    return wrapper


##########################################  |
#### Static kernel k : R^d x R^d -> R ####  |
########################################## \|/


class StaticKernel():
    """Static kernel k : R^d x R^d -> R."""

    def __call__(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool = False,
        )->Tensor:
        """
        Computes the Gram matrix k(X_i, Y_j), or the diagonal k(X_i, Y_i) 
        if diag=True, with batch support in the middle dimension. If X and Y
        are of ndim=1, they are reshaped to (1, d) and (1, d) respectively.

        Args:
            X (Tensor): Tensor of shape (... , d) or (N1, ... , d).
            Y (Tensor): Tensor of shape (... , d) or (N2, ... , d), 
                with (...) same as X.
            diag (bool): If True, only computes the kernel for
                the diagonal pairs k(X_i, Y_i). Defaults to False.
        
        Returns:
            Tensor: Tensor of shape (N1, N2, ...) or (N1, ...) if diag=True.
        """
        if X.ndim==1:
            X = X.unsqueeze(0)
        if Y.ndim==1:
            Y = Y.unsqueeze(0)
        return self._gram(X, Y, diag)
    

    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        )->Tensor:
        """
        Method to be implemented by subclasses. Computes the Gram matrix 
        k(X_i, Y_j), or the diagonal k(X_i, Y_i) if diag=True.

        Args:
            X (Tensor): Tensor with shape (N1, ... , d).
            Y (Tensor): Tensor with shape (N2, ... , d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs k(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2, ...) or (N1, ...) if diag=True.
        """
        raise NotImplementedError("Subclasses must implement '_gram' method.")
    

    def time_gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        )->Tensor:
        """
        Outputs k(X^i_s, Y^j_t), with optional diagonal support across
        the batch dimension.

        Args:
            X (Tensor): Tensor with shape (N1, T1, d).
            Y (Tensor): Tensor with shape (N2, T2, d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs k(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2, T1, T2) or (N1, T1, T2) if diag=True.
        """
        if diag:
            X = X.permute(1, 0, 2)
            Y = Y.permute(1, 0, 2)
            trans_gram = self(X, Y) # shape (T1, T2, N)
            return trans_gram.permute(2, 0, 1)
        else:
            N1, T1, d = X.shape
            N2, T2, d = Y.shape
            X = X.reshape(-1, d)
            Y = Y.reshape(-1, d)
            flat_gram = self(X, Y) # shape (N1 * T1, N2 * T2)
            gram = flat_gram.reshape(N1, T1, N2, T2)
            return gram.permute(0, 2, 1, 3)
    

    def squared_dist(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool = False,
        )->Tensor:
        """
        Computes the squared distance matrix between X and Y, i.e. 
        ||X_i - Y_j||^2, or the diagonal ||X_i - Y_i||^2 if diag=True,
        where ||.|| is the norm induced by the kernel.

        Args:
            X (Tensor): Tensor with shape (N1, ..., d).
            Y (Tensor): Tensor with shape (N2, ..., d), with (...) same as X.
            diag (bool, optional): If True, only computes the kernel for the

        Returns:
            Tensor: Tensor with shape (N1, N2, ...) or (N1, ...) if diag=True.
        """
        if diag:
            diff = X-Y
            norms_squared = self(diff, diff, diag=True) #shape (N1, ...)
        else:
            xx = self(X, X, diag=True) #shape (N1, ...)
            xy = self(X, Y, diag=False) #shape (N1, N2, ...)
            yy = self(Y, Y, diag=True) #shape (N2, ...)
            norms_squared = -2*xy + xx[:, None] + yy[None, :]

        return norms_squared
    

    def time_square_dist(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool = False,
        )->Tensor:
        """
        Outputs ||X^i_s - Y^j_t||^2, with optional diagonal support across
        the batch dimension.

        Args:
            X (Tensor): Tensor with shape (N1, T1, d).
            Y (Tensor): Tensor with shape (N2, T2, d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs k(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2, T1, T2) or (N1, T1, T2) if diag=True.
        """
        N1, T1, d = X.shape
        N2, T2, d = Y.shape
        if diag:
            X = X.permute(1, 0, 2) # shape (T1, N1, d)
            Y = Y.permute(1, 0, 2) # shape (T2, N2, d)
            trans_gram = self(X, Y) # shape (T1, T2, N)
            return trans_gram.permute(2, 0, 1)
        else:
            X = X.reshape(N1 * T1, d)
            Y = Y.reshape(N2 * T2, d)
            norms_squared = self.squared_dist(X, Y) # shape (N1 * T1, N2 * T2)
            return norms_squared.reshape(N1,T1,N2,T2).permute(0, 2, 1, 3)



##################################################################  |
#### Time series kernels k : R^(T1 x d) x R^(T2 x d) -> (...) ####  |
################################################################## \|/
        

class TimeSeriesKernel():
    """Time series kernel k : R^(T x d) x R^(T x d) -> (...)"""
    def __init__(
            self,
            max_batch: int = 1000,
            normalize: bool = False,
        ):
        self.max_batch = max_batch
        self.normalize = normalize


    @property
    def log_space(self):
        return False


    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool,
        ):
        """
        Method to be implemented by subclass. Computes the Gram matrix 
        k(X_i, Y_j) for time series X_i and Y_j, or the diagonal k(X_i, Y_i)
        if diag=True.

        Args:
            X (Tensor): Tensor with shape (N1, T, d).
            Y (Tensor): Tensor with shape (N2, T, d).
            diag (bool): If True, only computes the diagonal k(X_i, Y_i).

        Returns:
            Tensor: Tensor with shape (N1, N2, ...) or (N1, ...) if diag=True,
             where  (...) is the dimension of the kernel output.
        """
        raise NotImplementedError("Subclasses must implement '_gram' method.")


    def _max_batched_gram(
            self,
            X: Tensor,
            Y: Tensor,
            diag: bool,
            max_batch: Optional[int],
            normalize: Optional[bool],
            n_jobs: int,
        ):
        """
        Computes the Gram matrix k(X_i, Y_j) for time series X_i and Y_j, 
        or the diagonal k(X_i, Y_i) if diag=True.

        Args:
            X (Tensor): Tensor with shape (N1, T, d) or (T,d).
            Y (Tensor): Tensor with shape (N2, T, d) or (T,d).
            diag (bool): If True, only computes the kernel for the pairs
                k(X_i, Y_i). Defaults to False.
            max_batch (Optional[int]): Sets the max batch size if not None, 
                else uses the default 'self.max_batch'.
            normalize (Optional[int]): If True and diag=False, the kernel is normalized 
                to have unit diagonal via  K(X, Y) = K(X, Y) / sqrt(K(X, X) * K(Y, Y)), 
                and if None defaults to 'self.normalize'.
            n_jobs (int): Number of parallel jobs to run in joblib.Parallel.
        
        Returns:
            Tensor: Tensor with shape (N1, N2, ...) or (N1, ...) if diag=True,
                where (...) is the dimension of the kernel output.
        """
        N1, T, d = X.shape
        N2, _, _ = Y.shape
        max_batch = max_batch if max_batch is not None else self.max_batch
        normalize = normalize if normalize is not None else self.normalize

        # split into batches
        split_X = torch.split(X, max_batch, dim=0)
        Y_max_batch = max(1, max_batch//N1) if not diag else max_batch
        split_Y = torch.split(Y, Y_max_batch, dim=0)
        split = zip(split_X, split_Y) if diag else itertools.product(split_X, split_Y)
        result = Parallel(n_jobs=n_jobs)(
            delayed(self._gram)(x, y, diag) for x,y in split
            )

        # reshape back
        if diag:
            result = torch.cat(result, dim=0)
        elif max_batch >= N1:
            result = torch.cat(result, dim=1)
        else:
            extra = result[0].shape[2:]
            result = torch.cat(result, dim=0).reshape( (N1, N2)+extra )
    
        # normalize
        if normalize:
            #Obtain the diagonals k(X,X) and K(Y,Y)
            if X is Y:
                diagonal = result if diag else torch.einsum('ii...->i...', result) #shape (N, ...)
                XX = diagonal
                YY = diagonal
            else:
                XX = self._max_batched_gram(X, X, True, max_batch, False, n_jobs) #shape (N1, ...)
                YY = self._max_batched_gram(Y, Y, True, max_batch, False, n_jobs) #shape (N2, ...)
            if not diag:
                XX = XX[:, None] #shape (N1, 1, ...)
                YY = YY[None, :] #shape (1, N2, ...)

            # Normalize with diagonals k(X,Y) = k(X,Y) / sqrt(k(X,X) * k(Y,Y))
            if self.log_space:
                result = result - 0.5*XX - 0.5*YY
            else:
                result = result / torch.sqrt(XX) / torch.sqrt(YY)

        return result


    @is_documented_by(_max_batched_gram)
    def __call__(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool = False,
            max_batch: Optional[int] = None,
            normalize: Optional[bool] = None,
            n_jobs: int = 1,
        )->Tensor:

        # Reshape
        if X.ndim==2:
            X = X.unsqueeze(0)
        if Y.ndim==2:
            Y = Y.unsqueeze(0)

        # Compute and exponentiate if in log space
        result = self._max_batched_gram(X, Y, diag, max_batch, normalize, n_jobs)
        if self.log_space:
            result = torch.exp(result)

        return result