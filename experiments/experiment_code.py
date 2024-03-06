import numpy as np
import sklearn.preprocessing
import sklearn.utils
import sklearn.metrics
import torch
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
import tslearn
import tslearn.metrics
import sigkernel
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.anomaly_distance import BaseclassConformanceScore
from models.kernels import linear_kernel_gram, rbf_kernel_gram, poly_kernel_gram
from models.kernels import pairwise_kernel_gram, integral_kernel_gram, sig_kernel_gram
from models.normalize_streams import normalize_streams


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
              p:int,
              b:float,):
    """Calculates the gram matrices for the polynomial kernel.
    Train and test are of shape (N1, T, d) and (N2, T, d)."""
    poly_ker = lambda X, Y : poly_kernel_gram(X, Y, p, b)
    return case_static(train, test, poly_ker)


def case_gak(train:List[np.ndarray],
                   test:List[np.ndarray], 
                   fixed_length = True,
                   n_jobs:int = 1,
                   verbose:bool = False,
                   gak_factor:float = 1.0,):
    """Calculates the gram matrices for the gak kernel.
    Train and test are lists of possibly variable length multidimension 
    time series of shape (T_i, d)."""
    if fixed_length:
        #pick sigma parameter according to GAK paper
        sigma = gak_factor * tslearn.metrics.sigma_gak(np.array(train))
        vv_gram = tslearn.metrics.cdist_gak(train, sigma=sigma, n_jobs=n_jobs)
        uv_gram = tslearn.metrics.cdist_gak(test, train, sigma=sigma, n_jobs=n_jobs)
    else:
        sigma=gak_factor * 30.0
        kernel = lambda s1, s2 : tslearn.metrics.gak(s1, s2, sigma)
        vv_gram = pairwise_kernel_gram(train, train, kernel, sym=True, 
                                    n_jobs=n_jobs, verbose=verbose)
        uv_gram = pairwise_kernel_gram(test, train, kernel, sym=False, 
                                    n_jobs=n_jobs, verbose=verbose)
    return vv_gram, uv_gram


def stream_to_torch(
        stream:np.ndarray, 
        ):
    """Transforms a raw stream of shape (T, d) into a torch tensor."""
    N_i, new_d = stream.shape
    tensor = torch.from_numpy(stream).reshape(1, N_i, new_d).detach()
    return tensor


def case_sig_pde(train:List[np.ndarray], 
                 test:List[np.ndarray], 
                 dyadic_order:int = 5,
                 static_kernel = sigkernel.LinearKernel(),
                 n_jobs:int = 1,
                 verbose:bool = False,
                ):
    """Calculates the signature kernel gram matrices of the train and test.
    Train and test are lists of possibly variable length multidimension 
    time series of shape (T_i, d)."""
    sig_kernel = sigkernel.SigKernel(static_kernel, int(dyadic_order) )
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
    """Calculates the truncated signature kernel gram matrices of the train and test.
    Train and test are lists of possibly variable length multidimension time
    series of shape (T_i, d)."""
    vv_gram = sig_kernel_gram(train, train, order, static_kernel, only_last, 
                              sym=True, n_jobs=n_jobs, verbose=verbose)
    uv_gram = sig_kernel_gram(test, train, order, static_kernel, only_last, 
                              sym=False, n_jobs=n_jobs, verbose=verbose)
    return vv_gram, uv_gram


def case_integral(
        train:List[np.ndarray], 
        test:List[np.ndarray],
        static_kernel_with_diag:Callable,
        n_jobs:int = 1,
        verbose:bool = False,
        ):
    """Calculates the integral kernel gram matrices of the train and test,
    given a static kernel on R^d. Train and test are lists of possibly 
    variable length multidimension time series of shape (T_i, d)."""
    vv_gram = integral_kernel_gram(train, train, static_kernel_with_diag, 
                                    sym=True, n_jobs=n_jobs, verbose=verbose)
    uv_gram = integral_kernel_gram(test, train, static_kernel_with_diag, 
                            n_jobs=n_jobs, verbose=verbose)
    return vv_gram, uv_gram


def calc_grams(train:List[np.ndarray], 
               test:List[np.ndarray],
               param_dict:Dict[str, Any], # name : value
               sig_kernel_only_last:bool = True, #used in cross validation code
               n_jobs:int = 1,
               verbose:bool = False,
               ):   
    """Calculates gram matrices <train, train>, <test, train> given a kernel.
    Train and test are lists of possibly variable length multidimension time 
    series of shape (T_i, d)"""
    #choose method based on kernel name
    T, d = train[0].shape[-2:]
    kernel_name = param_dict["kernel_name"]
    if kernel_name == "linear":
        return case_linear(train, test)
    
    elif kernel_name == "rbf":
        return case_rbf(train, test, param_dict["sigma"])
    
    elif kernel_name == "poly":
        return case_poly(train, test, param_dict["p"], 1.0 - T/3) #minus T/3 to account for time augmentation. Only matters for flattened poly kernel

    elif kernel_name == "gak":
        return case_gak(train, test, True, n_jobs, verbose, param_dict["gak_factor"])
    
    if kernel_name == "truncated sig":
        ker = lambda X, Y: linear_kernel_gram(X, Y, custom_factor = 1/d * param_dict["scale"])
        return case_truncated_sig(train, test, param_dict["order"], 
                                  ker, sig_kernel_only_last, 
                                  n_jobs, verbose)
    
    elif kernel_name == "truncated sig rbf":
        ker = lambda X, Y: rbf_kernel_gram(X, Y, param_dict["sigma"]) 
        return case_truncated_sig(train, test, param_dict["order"], 
                                  ker, sig_kernel_only_last, n_jobs, verbose)
    
    elif kernel_name == "signature pde rbf":
        return case_sig_pde(train, test,
                            static_kernel=sigkernel.RBFKernel(param_dict["sigma"]*d), 
                            n_jobs=n_jobs, verbose=verbose,
                            dyadic_order=param_dict["dyadic_order"])

    elif kernel_name == "integral rbf":
        ker = lambda X, Y, diag: rbf_kernel_gram(X, Y, param_dict["sigma"], diag)
        return case_integral(train, test, ker, n_jobs, verbose)

    elif kernel_name == "integral poly":
        ker = lambda X, Y, diag : poly_kernel_gram(X, Y, param_dict["p"], 1.0, diag)
        return case_integral(train, test, ker, n_jobs, verbose)
    
    else:
        raise ValueError("Invalid kernel name:", kernel_name)


def compute_aucs(distances_conf:np.ndarray,    
                 distances_mahal:np.ndarray,    
                 y_test:np.ndarray,             
                 class_to_test,):
    """Computes the AUC scores based on one vs rest anomaly distances.

    Args:
        distances_conf (np.ndarray): Conformance anomaly distances, shape (N,).
        distances_mahal (np.ndarray): Mahalanobis anomaly distances, shape (N,).
        y_test (np.ndarray): Ground truth class labels, shape (N,).
        class_to_test (Any): Class label corresponding to the normal class.

    Returns:
        np.ndarray: AUC scores for conformance and mahalanobis distances,
                    and for ROC and PR AUC. Shape (2,2).
    """
    # 2 methods (conf, mahal), 2 metrics (roc_auc, pr_auc)
    aucs = np.zeros( (2, 2) ) 

    # Calculate one vs rest AUC, weighted by size of class
    for idx_conf_mahal, distances in enumerate([distances_conf, distances_mahal]):
        
        #PR AUC requires minority class to be label 1.
        #ROC AUC is agnostic to class imbalance.
        distances = 1/(distances+0.001)
        ovr_labels = y_test == class_to_test

        roc_auc = sklearn.metrics.roc_auc_score(ovr_labels, distances)
        pr_auc = sklearn.metrics.average_precision_score(ovr_labels, distances)
        aucs[idx_conf_mahal, 0] = roc_auc
        aucs[idx_conf_mahal, 1] = pr_auc
    
    return aucs


def get_corpus_and_test(X_train:List[np.ndarray], 
                        y_train:np.ndarray, 
                        X_test:List[np.ndarray], 
                        class_to_test:Any, 
                        ):
    """Returns the corpus and test set for a single class
    specified by label 'class_to_test'."""
    # Get all samples of the current class
    idxs = np.where(y_train == class_to_test)[0]
    corpus = [X_train[k] for k in idxs]
    corpus, test = normalize_streams(np.array(corpus), X_test)
    return corpus, test


def run_single_kernel_single_label(
        X_train:List[np.ndarray],
        y_train:np.ndarray,
        X_test:List[np.ndarray], 
        y_test:np.ndarray,
        class_to_test:Any,
        param_dict:Dict[str, Any], # name : value
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
                                       class_to_test)
        vv_gram, uv_gram = calc_grams(corpus, test, param_dict, 
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
                    y_train:np.ndarray, 
                    X_test:List[np.ndarray], 
                    y_test:np.ndarray, 
                    unique_labels:np.ndarray, 
                    kernelwise_dict:Dict[str, Dict[str, Dict[str, Any]]], # kernel_name : label : param_dict
                    n_jobs:int = 1, 
                    verbose:bool = True,
                    ) -> Dict[str, np.ndarray]:
    """Runs all kernels for all classes and computes AUC scores.

    Args:
        X_train (List[np.ndarray]): List of time series of shape (T_i, d).
        y_train (np.array): 1-dim array of class labels.
        X_test (List[np.ndarray]): List of time series of shape (T_i, d).
        y_test (np.array): 1-dim array of class labels.
        unique_labels (np.array): Unique class labels.
        kernelwise_dict (Dict[str, Dict[str, Dict[str, Any]]]): Nested dict 
                        of kernel parameters. kernel_name : label : param_dict
        verbose (bool): If True, prints progress.

    Returns:
        Dict[str, np.ndarray]: Dictionary of AUC scores for all kernels.
    """
    kernel_results = {}
    for kernel_name, labelwise_dict in tqdm(kernelwise_dict.items()):
        print("Kernel:", kernel_name)
        # 2 methods (conf, mahal), 2 metrics (roc_auc, pr_auc), C classes
        aucs = np.zeros( (2, 2, len(unique_labels)) ) 
        for i, (label, param_dict) in enumerate(labelwise_dict.items()):
            #run model
            scores = run_single_kernel_single_label(X_train, y_train, 
                                    X_test, y_test, label, param_dict,
                                    n_jobs=n_jobs, verbose=verbose,
                                    SVD_threshold=1e-10, SVD_max_rank=param_dict["threshold"])
            aucs[:,:, i] = scores
        
        #update kernel results
        kernel_results[kernel_name] = np.mean(aucs, axis=2)
    return kernel_results


def print_dataset_stats(num_classes, d, T, N_train, N_test):
    print("Number of Classes:", num_classes)
    print("Dimension of path:", d)
    print("Length:", T)
    print("Train:", N_train)
    print("Test:", N_test)

    
if __name__ == "__main__":
    pass