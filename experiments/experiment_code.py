import numpy as np
import sklearn.metrics
import scipy
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
import sys
import os
from pathlib import Path

# Get the current notebook path
notebook_path = os.path.abspath("new_cv.ipynb")
sys.path.insert(0, str(Path(notebook_path).parent / 'models'))
from models.anomaly_distance import BaseclassAnomalyScore
from models.stream_transforms import normalize_streams
from models.stream_transforms import augment_time, add_basepoint_zero, I_visibility_transform, T_visibility_transform
from models.kernels import StaticKernel, TimeSeriesKernel
from models.kernels import LinearKernel, RBFKernel, PolyKernel
from models.kernels import TruncSigKernel, SigPDEKernel, StaticIntegralKernel, FlattenedStaticKernel, GlobalAlignmentKernel, ReservoirKernel
from models.kernels import RandomizedSigKernel
from models.kernels import sigma_gak



def calc_grams(corpus:Tensor, 
               test:Tensor,
               param_dict:Dict[str, Any], # name : value
               sig_kernel_only_last:bool = True, #used in cross validation code
               ):   
    """Calculates gram matrices <train, train>, <test, train> given
    a param_dict specifying the kernel and its parameters.
    
    Args:
        corpus (Tensor): Array of shape (N, T, d).
        test (Tensor): Array of shape (N2, T, d).
        param_dict (Dict[str, Any]): Dictionary of kernel parameters.
        sig_kernel_only_last (bool): If True, only uses last signature level
            in truncated signature kernel.
    
    Returns:
        Tuple[Tensor, Tensor]: Gram matrices <train, train>, <test, train>."""
    #choose method based on kernel name
    _, T, d = corpus.shape

    kernel_name = param_dict["kernel_name"]
    if kernel_name == "flat linear":
        ker = FlattenedStaticKernel(LinearKernel(scale = 1/d/T), 
                                    normalize=param_dict["normalize"])
    
    elif kernel_name == "flat rbf":
        ker = FlattenedStaticKernel(RBFKernel(np.sqrt(d*T)*param_dict["sigma"]), 
                                    normalize=param_dict["normalize"])
    
    elif kernel_name == "flat poly":
        ker = FlattenedStaticKernel(PolyKernel(scale = 1/d/T, c=param_dict["c"], p=param_dict["p"]),
                                    normalize=param_dict["normalize"])

    elif kernel_name == "integral rbf":
        ker = StaticIntegralKernel(RBFKernel(np.sqrt(d)*param_dict["sigma"]),
                                   normalize=param_dict["normalize"])

    elif kernel_name == "integral poly":
        ker = StaticIntegralKernel(PolyKernel(scale = 1/d, c=param_dict["c"], p=param_dict["p"]),
                                   normalize=param_dict["normalize"])
    
    elif kernel_name == "trunc sig linear":
        ker = TruncSigKernel(LinearKernel(scale = 1/d * param_dict["scale"]),
                             trunc_level=param_dict["order"], only_last=sig_kernel_only_last,
                             normalize=param_dict["normalize"])
    
    elif kernel_name == "trunc sig rbf":
        ker = TruncSigKernel(RBFKernel(np.sqrt(d)*param_dict["sigma"], scale=param_dict["scale"]),
                             trunc_level=param_dict["order"], only_last=sig_kernel_only_last,
                             normalize=param_dict["normalize"])
    
    elif kernel_name == "pde sig rbf":
        ker = SigPDEKernel(RBFKernel(np.sqrt(d)*param_dict["sigma"], scale=param_dict["scale"]),
                           dyadic_order=param_dict["dyadic_order"],
                           normalize=param_dict["normalize"])
        
    elif kernel_name == "gak":
        ker = GlobalAlignmentKernel(RBFKernel(sigma_gak(corpus) * param_dict["gak_factor"]),
                                    normalize=param_dict["normalize"])
    
    elif kernel_name == "reservoir":
        # Reservoir kernel requires bounded inputs
        ker = ReservoirKernel(param_dict["tau"] / np.sqrt(d), param_dict["gamma"],
                              normalize=param_dict["normalize"])
        c = 1/param_dict["tau"]
        eps = 0.1
        corpus = torch.clamp(corpus, -c+eps, c-eps)
        test = torch.clamp(test, -c+eps, c-eps)
    
    elif "rand sig" in kernel_name:
        ker = RandomizedSigKernel(n_features= param_dict["n_features"], 
                               activation = param_dict["activation"],
                               seed = param_dict["seed"],
                               normalize=param_dict["normalize"])
        ker._init_given_input(corpus)
        ker.A /= np.sqrt(param_dict["scale"])
    
    torch.cuda.empty_cache()
    vv_gram = ker(corpus, corpus)
    torch.cuda.empty_cache()
    uv_gram = ker(test, corpus)
    return vv_gram, uv_gram



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



def get_corpus_and_test(X_train:Tensor, 
                        y_train:np.ndarray, 
                        X_test:Tensor, 
                        class_to_test:Any,
                        param_dict:Dict[str, Any], # name : value
                        ):
    """Returns the corpus and test set for a single class
    specified by label 'class_to_test'."""
    # Get all samples of the current class
    idxs = np.where(y_train == class_to_test)[0]
    corpus = X_train[idxs]
    corpus, test = normalize_streams(corpus, X_test)

    # Apply basepoint transform
    if param_dict["basepoint"] == "basepoint":
        corpus = add_basepoint_zero(corpus)
        test = add_basepoint_zero(test)
    elif param_dict["basepoint"] == "I_visibility":
        corpus = I_visibility_transform(corpus)
        test = I_visibility_transform(test)
    elif param_dict["basepoint"] == "T_visibility":
        corpus = T_visibility_transform(corpus)
        test = T_visibility_transform(test)
    
    # Apply time transform
    if param_dict["time"] == "time_enhance":
        corpus = augment_time(corpus)
        test = augment_time(test)

    return corpus, test



def run_single_kernel_single_label(
        X_train:Tensor,
        y_train:np.ndarray,
        X_test:Tensor, 
        y_test:np.ndarray,
        class_to_test:Any,
        param_dict:Dict[str, Any], # name : value
        SVD_threshold:float = 1e-10,
        SVD_max_rank:int = 50,
        verbose:bool = False,
        vv_gram:Optional[Tensor] = None,
        uv_gram:Optional[Tensor] = None,
        alphas:Optional[np.ndarray] = None,
        ):
    """Computes the AUC scores (one vs rest) for a single kernel,
    using kernelized nearest neighbour variance adjusted distances
    and mahalanobis distance.

    Args:
        X_train (Tensor): Array of shape (N, T, d).
        y_train (np.ndarray): 1-dim array of class labels.
        X_test (Tensor): Array of shape (N2, T, d)
        y_test (np.ndarray): 1-dim array of class labels.
        class_to_test (Any): Class label to test as normal class.
        param_dict (Dict[str, Any]): Dictionary of kernel parameters.
        SVD_threshold (float): Sets all eigenvalues below this threshold to be 0.
        verbose (bool): If True, prints progress.
        vv_gram (Tensor): Precomputed gram matrix for train set.
        uv_gram (Tensor): Precomputed gram matrix for test train pairs.
        alphas (Optional[nd.array]): If not None, then returns the AUCs
            corresponding to the smoothing parameter alpha in "param_dict",
            otherwise calculates the AUCs for all the alphas in alphas.
    
    Returns:
        np.ndarray: AUC scores for conformance and mahalanobis distances,
                    and for ROC and PR AUC. Shape (2,2), or (alphas, num_eigen, 2, 2)
                    if alphas is not None.
    """

    # Calculate amomaly distancce scores for all test samples
    if (vv_gram is None) or (uv_gram is None):
        corpus, test = get_corpus_and_test(X_train, y_train, X_test, 
                                       class_to_test, param_dict)
        vv_gram, uv_gram = calc_grams(corpus, test, param_dict)
    scorer = BaseclassAnomalyScore(vv_gram, SVD_threshold, SVD_max_rank, verbose=verbose)
        
    # only return accs for highest allowed threshold
    if (alphas is None):
        conf, mahal = scorer._anomaly_distance(uv_gram, method="both", alpha=param_dict["alpha"])
        conf = conf.cpu().numpy()
        mahal = mahal.cpu().numpy() #shape (N2,)
        aucs = compute_aucs(conf, mahal, y_test, class_to_test)
    else: #only used in cross validation code
        aucs = []
        for alpha in alphas:
            conf, mahal = scorer._anomaly_distance(uv_gram, method="both",
                                                    return_all_levels=True,
                                                    alpha=alpha)
            conf = conf.cpu().numpy()
            mahal = mahal.cpu().numpy() #shape (N2, num_eigenvalues)
            aucs_threshs = np.array([compute_aucs(c, m, y_test, class_to_test)
                            for c,m in zip(conf.T, mahal.T)]) #shape (num_eigenvalues, 2, 2)
            #pad and smooth a tiny bit
            aucs_threshs = np.pad(aucs_threshs, ((0, SVD_max_rank - aucs_threshs.shape[0]), (0,0), (0,0)) )
            aucs.append(aucs_threshs)
        aucs = np.array(aucs)
    return aucs #shape (2, 2) or (alphas, num_eigenvalues, 2, 2)



def run_all_kernels(X_train:Tensor, 
                    y_train:Tensor, 
                    X_test:Tensor, 
                    y_test:Tensor, 
                    unique_labels:Tensor, 
                    kernelwise_dict:Dict[str, Dict[str, Dict[str, Any]]], # kernel_name : label : param_dict
                    verbose:bool = True,
                    ) -> Dict[str, Tensor]:
    """Runs all kernels for all classes and computes AUC scores.

    Args:
        X_train (Tensor): Tensor of shape (N1 T, d).
        y_train (np.ndarray): 1-dim array of class labels.
        X_test (Tensor): Tensor of shape (N2, T, d).
        y_test (np.ndarray): 1-dim array of class labels.
        unique_labels (np.ndarray): Unique class labels.
        kernelwise_dict (Dict[str, Dict[str, Dict[str, Any]]]): Nested dict 
                        of kernel parameters. kernel_name : label : param_dict
        verbose (bool): Whether to print progress.

    Returns:
        Dict[str, Tensor]: Dictionary of AUC scores for all kernels.
    """
    kernel_results = {}
    for kernel_name, labelwise_dict in kernelwise_dict.items():
        print("Kernel:", kernel_name)
        # 2 methods (conf, mahal), 2 metrics (roc_auc, pr_auc), C classes
        aucs = np.zeros( (2, 2, len(unique_labels)) ) 
        for i, (label, param_dict) in enumerate(labelwise_dict.items()):
            #run model
            scores = run_single_kernel_single_label(X_train, 
                                    y_train, X_test, y_test, 
                                    label, param_dict,
                                    verbose=verbose, 
                                    SVD_max_rank=param_dict["threshold"])
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