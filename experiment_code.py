import numpy as np
import sklearn.preprocessing
import sklearn.utils
import sklearn.metrics
import torch
#from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
#from joblib import Memory, Parallel, delayed
import tslearn
import tslearn.metrics
from tslearn.datasets import UCR_UEA_datasets
import sigkernel

from conformance import BaseclassConformanceScore, stream_to_torch
from kernels import linear_kernel_gram, rbf_kernel_gram, poly_kernel_gram
from kernels import pairwise_kernel_gram, integral_kernel_gram, sig_kernel_gram



def print_dataset_stats(num_classes, d, T, N_train, N_test):
    print("Number of Classes:", num_classes)
    print("Dimension of path:", d)
    print("Length:", T)
    print("Train:", N_train)
    print("Test:", N_test)


def case_static(train:np.ndarray, 
                test:np.ndarray,
                static_kernel_gram:Callable,):
    """Calculates the gram matrices of equal length time series for 
    a static kernel on R^d. Train and test are of shape (N1, T, d) 
    and (N2, T, d). Static kernel should take in two arrays of shape 
    (M, T*d) and return the Gram matrix."""
    N1, T, d = train.shape
    N2, _, _ = test.shape
    train = train.reshape(N1, -1)
    test = test.reshape(N2, -1)
    vv_gram = static_kernel_gram(train, train)
    uv_gram = static_kernel_gram(test, train)
    return vv_gram, uv_gram


def case_linear(train:np.ndarray, 
                test:np.ndarray):
    """Calculates the gram matrices for the euclidean inner product.
    Train and test are of shape (N1, T, d) and (N2, T, d)."""
    return case_static(train, test, linear_kernel_gram)


def case_rbf(train:np.ndarray, 
             test:np.ndarray,
             sigma:float):
    """Calculates the gram matrices for the rbf kernel.
    Train and test are of shape (N1, T, d) and (N2, T, d)."""
    rbf_ker = lambda X, Y : rbf_kernel_gram(X, Y, sigma)
    return case_static(train, test, rbf_ker)


def case_poly(train:np.ndarray, 
              test:np.ndarray,
              p:float):
    """Calculates the gram matrices for the rbf kernel.
    Train and test are of shape (N1, T, d) and (N2, T, d)."""
    poly_ker = lambda X, Y : poly_kernel_gram(X, Y, p)
    return case_static(train, test, poly_ker)


def case_gak(train:List[np.ndarray],
                   test:List[np.ndarray], 
                   fixed_length:bool,
                   n_jobs:int = 1,
                   verbose:bool = False,
                   sigma:float = 1.0,):
    """Calculates the gram matrices for the gak kernel.
    Train and test are lists of possibly variable length multidimension 
    time series of shape (T_i, d)"""
    #pick sigma parameter according to GAK paper
    if fixed_length:
        sigma = tslearn.metrics.sigma_gak(np.array(train))

    #compute gram matrices
    kernel = lambda s1, s2 : tslearn.metrics.gak(s1, s2, sigma)
    vv_gram = pairwise_kernel_gram(train, train, kernel, sym=True, 
                                   n_jobs=n_jobs, verbose=verbose)
    uv_gram = pairwise_kernel_gram(test, train, kernel, sym=False, 
                                   n_jobs=n_jobs, verbose=verbose)
    return vv_gram, uv_gram


# Solely to be used in sigkernel library. See e.g. sigkernel.LinearKernel.
# Had to reimplement it since the original class is missing the scalar in 
# the Gram method
class LinearKernel():
    def __init__(self, scale=1.0):
        self.scale = scale
        
    def batch_kernel(self, X, Y):
        return self.scale*torch.bmm(X, Y.permute(0,2,1))

    def Gram_matrix(self, X, Y):
        return self.scale * torch.einsum('ipk,jqk->ijpq', X, Y)
    
class PolyKernel():
    def __init__(self, scale=1.0, p=2):
        self.scale = scale
        self.p = p
        
    def batch_kernel(self, X, Y):
        return self.scale * (1+torch.bmm(X, Y.permute(0,2,1)))**self.p

    def Gram_matrix(self, X, Y):
        return self.scale * (1+torch.einsum('ipk,jqk->ijpq', X, Y))**self.p

 
def case_sig_pde(train:List[np.ndarray], 
                 test:List[np.ndarray], 
                 dyadic_order:int = 3,
                 static_kernel = sigkernel.LinearKernel(),
                 n_jobs:int = 1,
                 verbose:bool = False,
                ):
    """Calculates the signature kernel gram matrices of the train and test.
    Train and test are lists of possibly variable length multidimension 
    time series of shape (T_i, d)"""
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)
    kernel = lambda s1, s2 : sig_kernel.compute_kernel(
                                stream_to_torch(s1), 
                                stream_to_torch(s2)).numpy()[0]
    vv_gram = pairwise_kernel_gram(train, train, kernel, sym=True, n_jobs=n_jobs, verbose=verbose)
    uv_gram = pairwise_kernel_gram(test, train, kernel, sym=False, n_jobs=n_jobs, verbose=verbose)
    return vv_gram, uv_gram


def case_truncated_sig(
        train:List[np.ndarray], 
        test:List[np.ndarray],
        order:int,
        static_kernel:Callable,
        only_last:bool,
        n_jobs:int = 1,
        verbose:bool = False,
        ):
    vv_gram = sig_kernel_gram(train, train, order, static_kernel, only_last, 
                              sym=True, n_jobs=n_jobs, verbose=verbose)
    uv_gram = sig_kernel_gram(test, train, order, static_kernel, only_last, 
                              sym=False, n_jobs=n_jobs, verbose=verbose)
    return vv_gram, uv_gram


def case_integral(
        train:List[np.ndarray], 
        test:List[np.ndarray],
        static_kernel_with_diag:Callable,
        fixed_length:bool,
        n_jobs:int = 1,
        verbose:bool = False,
        ):
        vv_gram = integral_kernel_gram(train, train, static_kernel_with_diag, 
                                fixed_length, sym=True, n_jobs=n_jobs, verbose=verbose)
        uv_gram = integral_kernel_gram(test, train, static_kernel_with_diag, 
                                fixed_length, n_jobs=n_jobs, verbose=verbose)
        return vv_gram, uv_gram


def calc_grams(train:List[np.ndarray], 
               test:List[np.ndarray],
               param_dict:Dict[str, Any], # name : value
               fixed_length:bool, 
               sig_kernel_only_last:bool = True, #used in cross validation code
               n_jobs:int = 1,
               verbose:bool = False,
               ):   
    """Calculates gram matrices <train, train>, <test, train> given a kernel.
    Train and test are lists of possibly variable length multidimension time 
    series of shape (T_i, d)"""

    #Transform to array if possible
    if fixed_length:
        train = np.array(train)
        test = np.array(test)
    
    #choose method based on kernel name
    kernel_name = param_dict["kernel_name"]
    if kernel_name == "linear":
        return case_linear(train, test)
    
    elif kernel_name == "rbf":
        return case_rbf(train, test, param_dict["sigma"])
    
    elif kernel_name == "poly":
        return case_poly(train, test, param_dict["p"])

    elif kernel_name == "gak":
        return case_gak(train, test, fixed_length, n_jobs, verbose)

    elif kernel_name == "truncated sig":
        return case_truncated_sig(train, test, param_dict["order"], 
                                  linear_kernel_gram, sig_kernel_only_last, 
                                  n_jobs, verbose)
    
    elif kernel_name == "truncated sig rbf":
        ker = lambda X, Y: rbf_kernel_gram(X, Y, param_dict["sigma"])
        return case_truncated_sig(train, test, param_dict["order"], 
                                  ker, sig_kernel_only_last, n_jobs, verbose)
    
    elif kernel_name == "truncated sig poly":
        ker = lambda X, Y : poly_kernel_gram(X, Y, param_dict["p"])
        return case_truncated_sig(train, test, param_dict["order"], 
                                  ker, sig_kernel_only_last, n_jobs, verbose)
    
    elif kernel_name == "signature pde":
        return case_sig_pde(train, test,
                            static_kernel=LinearKernel(1/train[0].shape[-1]),
                            n_jobs=n_jobs, verbose=verbose)
    
    elif kernel_name == "signature pde rbf":
        return case_sig_pde(train, test,
                            static_kernel=sigkernel.RBFKernel(
                                param_dict["sigma"] * train[0].shape[-1]),
                            n_jobs=n_jobs, verbose=verbose)

    elif kernel_name == "signature pde poly":
        return case_sig_pde(train, 
                            test,
                            static_kernel=PolyKernel(
                                1/train[0].shape[-1], param_dict["p"]),
                            n_jobs=n_jobs, verbose=verbose)
    
    elif kernel_name == "integral linear":
        return case_integral(train, test, linear_kernel_gram, fixed_length, n_jobs, verbose)

    elif kernel_name == "integral rbf":
        ker = lambda X, Y, diag: rbf_kernel_gram(X, Y, param_dict["sigma"], diag)
        return case_integral(train, test, ker, fixed_length, n_jobs, verbose)

    elif kernel_name == "integral poly":
        ker = lambda X, Y, diag : poly_kernel_gram(X, Y, param_dict["p"], diag)
        return case_integral(train, test, ker, fixed_length, n_jobs, verbose)
    
    else:
        raise ValueError("Invalid kernel name:", kernel_name)


def normalize_streams(train:np.ndarray, 
                      test:np.ndarray,
                      ):
    """Inputs are 3D arrays of shape (N, T, d) where N is the number of time series, 
    T is the length of each time series, and d is the dimension of each time series."""
    # Normalize data by training set mean and std
    mean = np.mean(train, axis=0, keepdims=True)
    std = np.std(train, axis=0, keepdims=True)
    train = (train - mean) / std
    test = (test - mean) / std
    return train, test


def compute_aucs(distances_conf:np.ndarray,     #size N
                 distances_mahal:np.ndarray,    #size Nint
                 y_test:np.array,               #size N
                 class_to_test,):
    # 2 methods (conf, mahal), 2 metrics (roc_auc, pr_auc)
    aucs = np.zeros( (2, 2) ) 

    # Calculate one vs rest AUC, weighted by size of class
    for idx_conf_mahal, distances in enumerate([distances_conf, distances_mahal]):
        ovr_labels = y_test != class_to_test
        average="weighted" #average = "macro" or "weighted"
        roc_auc = sklearn.metrics.roc_auc_score(ovr_labels, distances, average=average)
        pr_auc = sklearn.metrics.average_precision_score(ovr_labels, distances, average=average)
        aucs[idx_conf_mahal, 0] = roc_auc
        aucs[idx_conf_mahal, 1] = pr_auc
    
    return aucs


def get_corpus_and_test(X_train:List[np.ndarray], 
                        y_train:np.array, 
                        X_test:List[np.ndarray], 
                        class_to_test:int, 
                        fixed_length:bool,):
    # Get all samples of the current class
    idxs = np.where(y_train == class_to_test)[0]
    corpus = [X_train[k] for k in idxs]
    test = X_test
    if fixed_length:
        corpus, test = normalize_streams(np.array(corpus), test)
    return corpus, test


def run_single_kernel_single_label(
        X_train:List[np.ndarray],
        y_train:np.ndarray,
        X_test:List[np.ndarray], 
        y_test:np.ndarray,
        class_to_test:Any,
        param_dict:Dict[str, Any], # name : value
        fixed_length:bool,
        SVD_threshold:float = 10e-14,
        SVD_max_rank:Optional[int] = None,
        verbose:bool = False,
        vv_gram:Optional[np.ndarray] = None,
        uv_gram:Optional[np.ndarray] = None,
        return_all_levels:bool = False,
        n_jobs:int = 1, 
        ):
    """Computes the AUC scores (weighted one vs rest) for a single kernel,
    using kernelized nearest neighbour variance adjusted distances.

    Args:
        X_train (List[np.ndarray]): List of time series of shape (T_i, d).
        y_train (np.array): 1-dim array of class labels.
        X_test (List[np.ndarray]): List of time series of shape (T_i, d).
        y_test (np.array): 1-dim array of class labels.
        class_to_test (Any): Class label to test as normal class.
        param_dict (Dict[str, Any]): Dictionary of kernel parameters.
        fixed_length (bool): If True, uses the optimized kernels for fixed 
                             length time series.
        SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
        SVD_max_rank (int): Sets all SVD eigenvalues to be 0 beyond 'SVD_max_rank'.
        verbose (bool): If True, prints progress.
        vv_gram (np.ndarray): Precomputed gram matrix for train set.
        uv_gram (np.ndarray): Precomputed gram matrix for test train pairs.
        return_all_levels (bool): If True, returns AUCs for all levels of the
                                    anomaly distance scores.
        n_jobs (int): Number of parallel jobs to run in pairwise Gram calculations.
    """

    # Calculate amomaly distancce scores for all test samples
    if (vv_gram is None) and (uv_gram is None):
        corpus, test = get_corpus_and_test(X_train, y_train, X_test, 
                                       class_to_test, fixed_length)
        vv_gram, uv_gram = calc_grams(corpus, test, param_dict, fixed_length, 
                                      n_jobs=n_jobs, verbose=verbose)
    scorer = BaseclassConformanceScore(vv_gram, SVD_threshold, verbose=verbose, 
                                    SVD_max_rank=SVD_max_rank)
        
    # only return accs for highest allowed threshold
    if not return_all_levels:
        conf, mahal = scorer._anomaly_distance(uv_gram, method="both")
        aucs = compute_aucs(conf, mahal, y_test, class_to_test)
    else:
        conf, mahal = scorer._anomaly_distance(uv_gram, method="both",
                                                return_all_levels=True)
        aucs = np.array([compute_aucs(c, m, y_test, class_to_test)
                         for c,m in zip(conf.T, mahal.T)])

    return aucs #shape (2, 2) or (M, 2, 2)



def run_all_kernels(X_train:List[np.ndarray], 
                    y_train:np.array, 
                    X_test:List[np.ndarray], 
                    y_test:np.array, 
                    unique_labels:np.array, 
                    kernelwise_dict:Dict[str, Dict[str, Dict[str, Any]]], # kernel_name : label : param_dict
                    fixed_length:bool,
                    n_jobs:int = 1, 
                    verbose:bool = True,
                    ):
    kernel_results = {}
    for kernel_name, labelwise_dict in kernelwise_dict.items():
        # 2 methods (conf, mahal), 2 metrics (roc_auc, pr_auc), C classes
        aucs = np.zeros( (2, 2, len(unique_labels)) ) 
        for i, (label, param_dict) in enumerate(labelwise_dict.items()):
            #run model
            scores = run_single_kernel_single_label(X_train, y_train, 
                                    X_test, y_test, label, param_dict,
                                    fixed_length, n_jobs=n_jobs, verbose=verbose,
                                    SVD_threshold=0, SVD_max_rank=param_dict["threshold"])
            aucs[:,:, i] = scores
        
        #update kernel results
        kernel_results[kernel_name] = np.mean(aucs, axis=2)
    return kernel_results



def validate_tslearn(
        dataset_kernel_label_paramdict : Dict[str, Dict[str, Dict[str, Any]]],
        n_jobs:int = 1, 
        verbose:bool=True,
        ):
    """Validates the best models from cross validation on the
    tslearn datasets using kernel conformance scores."""
    experiments = {}
    for dataset_name, results in dataset_kernel_label_paramdict.items():

        # Load dataset
        print(dataset_name)
        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)
        N_train, T, d = X_train.shape
        N_test, _, _  = X_test.shape
        print_dataset_stats(num_classes, d, T, N_train, N_test)

        #validate on test set
        kernelwise_dict = results["kernel_results"]
        kernel_results = run_all_kernels(X_train, y_train, X_test, y_test, 
                            unique_labels, kernelwise_dict, fixed_length=True, 
                            n_jobs=n_jobs, verbose=verbose)
        experiments[dataset_name] = {"results": kernel_results, 
                                     "num_classes": num_classes, 
                                     "path dim":d,
                                     "ts_length":T, 
                                     "N_train":N_train, 
                                     "N_test":N_test}
    return experiments


if __name__ == "__main__":
    pass
    #read in the results from cross validations and validate on test set