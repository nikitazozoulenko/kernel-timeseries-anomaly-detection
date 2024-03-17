import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
from tslearn.datasets import UCR_UEA_datasets
import torch
from torch import Tensor
from sklearn.model_selection import RepeatedStratifiedKFold

import pickle
import time
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_code import run_single_kernel_single_label, get_corpus_and_test
from experiments.experiment_code import calc_grams, print_dataset_stats
from experiments.utils import save_to_pickle


#######################################################################
################### Hyperparameter combinations #######################
#######################################################################


def str_to_original(val):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            if val== 'True':
                return True
            elif val == 'False':
                return False
            else:
                return val

def get_hyperparam_ranges(kernel_name:str):
    """ Returns a dict of hyperparameter ranges for the specified kernel."""

    # Dict for hyperparameter ranges
    ranges = {}

    # Stream transforms for all kernels
    ranges["basepoint"] = ["", "basepoint", "I_visibility", "T_visibility"]
    ranges["time"] = ["", "time_enhance"]

    # Specific to each state-space kernel
    if "rbf" in kernel_name:
        ranges["sigma"] = np.exp(np.linspace(-2, 3, 7))
    elif "poly" in kernel_name:
        ranges["p"] = np.array([2, 3, 4, 5, 6])

    # Specific to each time series kernel
    if "gak" in kernel_name:
        ranges["gak_factor"] = np.exp(np.linspace(-2, 2, 5))
    elif "pde" in kernel_name:
        ranges["dyadic_order"] = np.array([3], dtype=np.int64)
    elif "reservoir" in kernel_name:
        ranges["tau"] = np.array([1/6]) #recall that we clipped by 5
        ranges["gamma"] = np.exp(np.linspace(-4, -0.1, 20))
    
    #For trunc sig we get all orders up to MAX_ORDER for free
    if "trunc sig" in kernel_name:
        MAX_ORDER = 10
        ranges["order"] = np.array([MAX_ORDER])
        ranges["scale"] = np.array([1/2, 1, 2]) #TODO add to pde sig too

    #For all kernels, we can normalize or not
    if "gak" in kernel_name or "pde sig linear" in kernel_name:
        ranges["normalize"] = np.array([True])
    else:
        ranges["normalize"] = np.array([True, False])

    return ranges



def get_hyperparam_combinations(kernel_name:str):
    """Returns a list of param_dicts for the specified kernel."""
    ranges = get_hyperparam_ranges(kernel_name)
    values = ranges.values()
    keys = ranges.keys()

    if ranges:
        #create array of all combinations
        meshgrid = np.meshgrid(*values)
        combinations = np.vstack([x.flatten() for x in meshgrid]).T
        
        #convert to dict
        dict_combinations = [dict(zip(keys, [str_to_original(val) for val in vals])) for vals in combinations]
        return dict_combinations
    else:
        return [{}]


#######################################################################
######################### Cross Validation ############################
#######################################################################


def aucs_to_objective(aucs:np.ndarray): #shape (M, 2, 2)
    """Takes AUCs from 'run_single_kernel_single_label' and outputs the
    objective for Cross Validation for both Conf and Mahal.
    
    Args:
        aucs (np.ndarray): AUC scores for conformance and mahalanobis distances,
                    and for ROC and PR AUC, for each alpha and truncation level.
                    Shape (len(alphas), len(thresholds), 2, 2).
    
    Returns:
        np.ndarray: Objective for Cross Validation for both Conf and Mahal.
                    Shape (len(alphas), len(thresholds), 2)."""
    aucs = np.sum(aucs, axis=-1) # sum of ROC AUC and PR AUC
    return aucs #shape (len(alphas), len(threhsolds), 2)



def eval_1_paramdict_1_fold(X_train, 
                            y_train, 
                            X_val, 
                            y_val,
                            class_to_test:Any,
                            param_dict:Dict[str, Any],
                            thresholds:np.ndarray,
                            alphas:np.ndarray,
                            verbose:bool=False,
                            ):
    """Evaluates a single fold for a single hyperparameter configuration.

    Args:
        X_train (Tensor): Tensor of shape (N_train, T, d).
        y_train (np.ndarray): 1-dim array of class labels.
        X_val (Tensor): Tensor of shape (N_val, T, d).
        y_val (np.ndarray): 1-dim array of class labels.
        class_to_test (Any): The normal class.
        param_dict (Dict[str, Any]): Hyperparameter configuration.
        thresholds (np.ndarray): Thresholds for covariance operator eigenvalues.
        verbose (bool): Verbosity.
    
    Returns:
        np.ndarray: Objective scores for conformance and mahalanobis distances,
                    shape (len(alphas), len(thresholds), 2), or
                    shape (len(alphas), len(thresholds), 2, n_truncs) for truncated sig.
    """
    SVD_threshold = np.min(thresholds)
    corpus, test = get_corpus_and_test(X_train, y_train, X_val, 
                                class_to_test, param_dict)
    vv_grams, uv_grams = calc_grams(corpus, test, param_dict, 
                                    sig_kernel_only_last=False)
    
    def get_objective(vv, uv):
        raw_aucs = run_single_kernel_single_label(X_train, y_train, X_val, y_val,
                            class_to_test, param_dict,
                            SVD_threshold, verbose,
                            vv, uv, thresholds, alphas)
        return aucs_to_objective(raw_aucs)
    

    # Simple case for most methods
    if "trunc sig" not in param_dict["kernel_name"]:
        objectives = get_objective(vv_grams, uv_grams)
    # Computationally efficient truncated signature case
    else:
        objectives = np.stack([get_objective(vv, uv) 
                         for vv, uv in zip(vv_grams.permute(2,0,1), 
                                           uv_grams.permute(2,0,1))],
                         axis=-1)
    return objectives # shape (len(alphas), len(thresholds), 2) or (len(alphas), len(thresholds), 2, n_truncs) for truncated sig



def eval_repeats_folds(X:Tensor,
                       y:np.ndarray,
                rskf:RepeatedStratifiedKFold,
                kernel_name:str,
                hyperparams:List[Dict[str, Any]],  #List of {name : hyperparam_value}
                class_to_test:Any,
                thresholds:np.ndarray,
                alphas:np.ndarray,
                verbose:bool = False,
                ):
    """Evaluates the performance of the given hyperparameters on the 
    given dataset using repeated k-fold cross-validation. Returns the
    AUC scores for each hyperparameter configuration.

    Args:
        X (Tensor): Tensor of shape (N, T, d).
        y (np.ndarray): 1-dim array of class labels.
        rskf (RepeatedStratifiedKFold): Sklearn fold generator.
        kernel_name (str): The name of the kernel.
        hyperparams (List[Dict[str, Any]]): List of hyperparameter configurations.
        class_to_test (Any): The normal class.
        verbose (bool): Verbosity.
    
    Returns:
        np.ndarray: Objectives scores for each hyperparameter configuration.
                    Shape (n_hyperparams, len(alphas), len(thresholds), 2), or
                    shape (n_hyperparams, len(alphas), len(thresholds), 2, n_truncs)
    """
    fold_indices = list(rskf.split(X,y))

    #for each parameter:
    scores = []
    for i, param_dict in enumerate(hyperparams):
        param_dict["kernel_name"] = kernel_name
        param_dict["normal_class_label"] = class_to_test

        #loop over repeats and folds
        folds = []
        for train_idx, val_idx in fold_indices:
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            obj = eval_1_paramdict_1_fold(
                            X_train, y_train, X_val, y_val,
                            class_to_test, param_dict, 
                            thresholds, alphas, verbose)
            folds.append(obj)

        scores.append(folds)

    #average across repeats and folds
    scores = np.array(scores) # shape (n_hyperparams, n_folds_repeats, len(alphas), len(thresholds), 2, (opt. dim: n_truncs))
    scores = np.mean(scores, axis=1)
    return scores #shape (n_hyperparams, len(alphas), len(thresholds), 2, (opt. dim: n_truncs))



def choose_best_hyperparam(scores_conf_mahal:np.ndarray,
                           hyperparams:List[Dict[str, Any]],
                           thresholds:np.ndarray,
                           alphas:np.ndarray,
                        ):
    """Chooses the best hyperparameter configuration based on the AUC 
    scores outputed from 'eval_repeats_folds'.
    
    Args:
        scores_conf_mahal (np.ndarray): AUC scores for each hyperparameter configuration
                                        outputed from 'eval_repeats_folds'.     
        hyperparams (List[Dict[str, Any]]): List of param_dicts.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Best hyperparameter configuration
                    for conformance and mahalanobis distances.
    """
    c_m_param_dicts = [{}, {}]
    for i in range(2):
        #scores_conf_mahal shape (n_hyperparams, alphas, thresholds, 2, (opt. dim: n_truncs))
        scores = scores_conf_mahal[..., i] # shape (n_hyperparams, alphas, thresholds, (opt. dim: n_truncs))
        dims = np.arange(scores.ndim) 
        max_params = np.max(scores, axis=tuple(dims[1:]) )
        best_param_idx = np.argmax(max_params)
        max_alpha = np.max(scores, axis=(0, *dims[2:]))
        best_alpha_idx = np.argmax(max_alpha)
        max_thresh = np.max(scores, axis=(0, 1,  *dims[3:]))
        best_thresh_idx = np.argmax(max_thresh)

        #choose best param_dict
        final_param_dict = hyperparams[best_param_idx].copy()
        final_param_dict["threshold"] = thresholds[best_thresh_idx]
        final_param_dict["alpha"] = alphas[best_alpha_idx]
        final_param_dict["CV_train_score"] = max_params[best_param_idx]
        kernel_name = final_param_dict["kernel_name"]

        #store some extra stats
        final_param_dict["score_thresh"] = max_thresh
        final_param_dict["score_alphas"] = max_alpha

        #optional: best truncation level
        if "trunc sig" in kernel_name:
            max_truncs = np.max(scores, axis=(0, 1, 2))
            best_trunc_idx = np.argmax(max_truncs)
            final_param_dict["order"] = 1+best_trunc_idx
            final_param_dict["score_truncations"] = max_truncs
        c_m_param_dicts[i] = final_param_dict

    return c_m_param_dicts



def cv_given_dataset(X:Tensor,                  #Training Dataset
                    y:np.ndarray,               #Training class labels
                    unique_labels:np.ndarray,   #Unique class labels
                    kernel_names:List[str],
                    k_folds:int,                #k-fold cross validation
                    n_repeats:int,              #repeats of k-fold CV
                    verbose:bool = False
                    ):
    """Performs repeated k-fold cross-validation on the given dataset 
    for the anomaly detection models specified by 'kernel_names'. We 
    use the AUC scores to evaluate the performance of the models. Saves
    the result as a nested dictionary of the form {kernel : label : params}"""

    rskf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_repeats)
    thresholds = np.exp(np.linspace(3, -10, 20))
    alphas = np.array([0, 10**-6, 10**-4, 10**-2])

    #store for conf and mahal separately
    c_kernelwise_param_dicts = {} # kernel : label : param_dict
    m_kernelwise_param_dicts = {}
    for kernel_name in kernel_names:
        hyperparams = get_hyperparam_combinations(kernel_name)

        #loop over normal class
        c_labelwise_param_dicts = {} # label : param_dict
        m_labelwise_param_dicts = {}
        t0 = time.time()
        for label in tqdm(unique_labels, desc = f"Label for {kernel_name}"):
            scores = eval_repeats_folds(X, y, rskf, kernel_name,
                                        hyperparams, label, thresholds,alphas, verbose)
            c_param_dict, m_param_dict = choose_best_hyperparam(scores, hyperparams, thresholds, alphas)
            c_labelwise_param_dicts[label] = c_param_dict
            m_labelwise_param_dicts[label] = m_param_dict
        c_kernelwise_param_dicts[kernel_name] = c_labelwise_param_dicts
        m_kernelwise_param_dicts[kernel_name] = m_labelwise_param_dicts

        #add elapsed CV time to each param_dict
        t1 = time.time()
        elapsed_time = t1-t0
        print(f"Time taken for kernel {kernel_name}:", elapsed_time, "seconds")
        for label in unique_labels:
            for c_or_m in [c_kernelwise_param_dicts, m_kernelwise_param_dicts]:
                c_or_m[kernel_name][label]["CV_time"] = elapsed_time

    return c_kernelwise_param_dicts, m_kernelwise_param_dicts



def cv_tslearn(dataset_names:List[str], 
                kernel_names:List[str],
                k_folds:int = 5,           # k-fold cross validation
                n_repeats:int = 1,         # repeats of k-fold CV)
                verbose:bool = False,
                device="cuda",
                ):    
    """Cross validation for tslearn datasets.
    
    Args:
        dataset_names (List[str]): List of tslearn dataset names.
        kernel_names (List[str]): List of kernel names.
        k_folds (int): k-fold cross validation.
        n_repeats (int): Repeats of k-fold CV.
        verbose (bool): Verbosity.
        device (str): Device for PyTorch computation.
        """
    cv_best_models = {} # dataset_name : kernel_name : label : param_dict
    for dataset_name in dataset_names:
        print("Dataset:", dataset_name)
        # Load dataset
        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
        X_train = torch.from_numpy(X_train).to(device)
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)
        N_train, T, d = X_train.shape
        print_dataset_stats(num_classes, d, T, N_train, "N/A")

        # Run each kernel
        t0 = time.time()
        c_kernelwise_param_dicts, m_kernelwise_param_dicts = cv_given_dataset(
                                                X_train, y_train, unique_labels, 
                                                kernel_names, k_folds, n_repeats,
                                                verbose)
        t1 = time.time()
        print(f"Time taken for dataset {dataset_name}:", t1-t0, "seconds\n\n\n")
        
        #log dataset experiment
        cv_best_models[dataset_name] = {
                                    "conf_results": c_kernelwise_param_dicts,
                                    "mahal_results": m_kernelwise_param_dicts, 
                                    "num_classes": num_classes, 
                                    "path dim":d,
                                    "ts_length":T, 
                                    "N_train":N_train
                                     }

    return cv_best_models


#######################################################################
######################### Print CV Results ############################
#######################################################################


def average_labels(labelwise_dict:Dict[str, Dict[str, Any]],
                    field:str):
    """Averages the values of a field over the labels."""
    L = [param_dict[field] for param_dict in labelwise_dict.values()]
    min_len = min([np.array(Li).size for Li in L])
    if min_len > 1:
        L = [Li[:min_len] for Li in L]
    return np.mean(L,axis=0)



def print_cv_results(
        dataset_kernel_label_paramdict : Dict[str, Dict[str, Dict[str, Any]]],
        ):
    """Prints the results of cross validation on the tslearn datasets
    given a dict of form {dataset_name : kernel_name : label : param_dict},
    outputed by 'cv_tslearn'."""

    with np.printoptions(precision=3, suppress=True):
        print("Cross Validation Results")
        for dataset_name, results in dataset_kernel_label_paramdict.items():
            print_dataset_stats(results['num_classes'], results['path dim'], 
                                results['ts_length'], results['N_train'], "N/A")
            for anomaly_method in ["conf_results", "mahal_results"]:
                print(f"\n{anomaly_method}")
                kernelwise_dict = results[anomaly_method]
                for kernel_name, labelwise_dict in kernelwise_dict.items():
                    final_score_avgs = average_labels(labelwise_dict, "CV_train_score")
                    alphas_score_avgs = average_labels(labelwise_dict, "score_alphas")
                    thresh_score_avgs = average_labels(labelwise_dict, "score_thresh")
                    print(f"\n{kernel_name}")
                    print("final_score_avgs", final_score_avgs)
                    print("alphas_score_avgs", alphas_score_avgs)
                    print("thresh_score_avgs", thresh_score_avgs)
                    if "truncated sig" in kernel_name:
                        trunc_score_avgs = average_labels(labelwise_dict, "score_orders")
                        print("orders_score_avgs", trunc_score_avgs)
                    
                    for label, param_dict in labelwise_dict.items():
                        print(label)
                        print({k:v for k,v in param_dict.items() 
                            if k not in ["kernel_name", "normal_class_label", 
                                            "CV_train_score", "score_alphas", "score_thresh", 
                                            "score_orders"]})
            print("\nEnd dataset \n\n\n")


#python3 experiments/cross_validation.py  --dataset_names "Epilepsy" --k_folds 5 --n_repeats 5 --save_path "Data/cv_Epilepsy.pkl"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run this script to run cross validation on ts-learn datasets.")
    parser.add_argument("--dataset_names", nargs="+", type=str, default=[
        'Epilepsy',                    # N_corpus = 34
        'EthanolConcentration',        # N_corpus = 65
        'FingerMovements',             # N_corpus = 158
        'HandMovementDirection',       # N_corpus = 40
        'Heartbeat',                   # N_corpus = 102
        'LSST',                        # N_corpus = 176 N_train = 3000 ish
        'MotorImagery',                # N_corpus = 139
        'NATOPS',                      # N_corpus = 30
        'PEMS-SF',                     # N_corpus = 38
        'PhonemeSpectra',              # N_corpus = 85 N_train = 3000 ish
        'RacketSports',                # N_corpus = 38
        'SelfRegulationSCP1',          # N_corpus = 134
        'SelfRegulationSCP2',          # N_corpus = 100
        ])
    parser.add_argument("--kernel_names", nargs="+", type=str, default=[
                "flat linear",
                "flat rbf",
                "flat poly",

                "integral rbf",
                "integral poly",

                "trunc sig linear",
                "trunc sig rbf",
                "pde sig rbf",

                "gak", #normalized only

                "reservoir",
                ])
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=f"Data/cv_{int(time.time()*1000)}.pkl")
    args = vars(parser.parse_args())
    print("Args:", args)

    cv_best_models = cv_tslearn(
            dataset_names = args["dataset_names"],
            kernel_names = args["kernel_names"],
            k_folds = args["k_folds"],
            n_repeats = args["n_repeats"],
                )
    
    #save to disk
    save_to_pickle(cv_best_models, args["save_path"])
    print_cv_results(cv_best_models)