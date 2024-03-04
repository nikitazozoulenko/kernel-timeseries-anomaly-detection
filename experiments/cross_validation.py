import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
from joblib import Memory, Parallel, delayed
import tslearn
import tslearn.metrics
from tslearn.datasets import UCR_UEA_datasets
import pickle
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_code import run_single_kernel_single_label, get_corpus_and_test
from experiments.experiment_code import calc_grams, print_dataset_stats
from experiments.utils import save_to_pickle


#######################################################################
########################## Repeated k-folds ###########################
#######################################################################


def repeat_k_folds(X:List,    #dataset
                y:np.ndarray, #class labels
                k:int,
                n_repeats:int,):
    repeats = [create_k_folds(X, y, k) for _ in range(n_repeats)]
    return repeats



def create_k_folds(X:List,    #dataset
                y:np.ndarray, #class labels
                k:int,):
    """Generates balanced k-folds for cross-validation, where each fold
    is balanced the same as the original dataset."""

    #is X numpy array?
    is_numpy=True if isinstance(X, np.ndarray) else False

    #shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = [X[i] for i in indices]
    y = np.array([y[i] for i in indices])
    unique_labels = np.unique(y)

    #split into classes
    classwise = {label:[] for label in unique_labels}
    for x, label in zip(X, y):
        classwise[label].append(x)

    #split into k-folds
    classwise_folds = {}
    for label, dataclass in classwise.items():
        classwise_folds[label] = np.array_split(dataclass, k)
    
    #create folds
    folds=[]
    for i in range(k):
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        for label, dataclass in classwise_folds.items():
            for j in range(k):
                if j!=i:
                    X_train.extend(dataclass[j])
                    y_train.extend([label]*len(dataclass[j]))
                else:
                    X_val.extend(dataclass[j])
                    y_val.extend([label]*len(dataclass[j]))

        #convert to numpy if possible
        if is_numpy:
            X_train = np.array(X_train)
            X_val = np.array(X_val)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        folds.append([X_train, y_train, X_val, y_val])

    return folds

#######################################################################
################### Hyperparameter combinations #######################
#######################################################################

def get_hyperparam_combinations(kernel_name:str):
    """Returns a dict of hyperparameter ranges and a list of all 
    possible combinations of hyperparameters for the specified kernel"""
    ranges = get_hyperparam_ranges(kernel_name)
    values = ranges.values()
    keys = ranges.keys()

    if ranges:
        #create array of all combinations
        meshgrid = np.meshgrid(*values)
        combinations = np.vstack([x.flatten() for x in meshgrid]).T

        #convert to dict
        dict_combinations = [dict(zip(keys, vals)) for vals in combinations]
        return dict_combinations
    else:
        return [{}]



def get_hyperparam_ranges(kernel_name:str):
    """ Returns a dict of hyperparameter ranges for the specified kernel."""
    max_poly_p = 5
    n_sigmas = 5
    ranges = {}
    dyadic_order = 2

    #static kernel params. Note that sig and integral kernels also use this
    if "rbf" in kernel_name:
        ranges["sigma"] = np.array([10**k for k in np.linspace(-3, 1, n_sigmas)])
    elif "poly" in kernel_name:
        ranges["p"] = np.arange(2, max_poly_p+1)
    elif "gak" in kernel_name:
        ranges["gak_factor"] = np.array([0.333, 1, 3])

    if "pde" in kernel_name:
        ranges["dyadic_order"] = np.array([dyadic_order], dtype=np.int64)

    return ranges
    
#######################################################################
######################### Cross Validation ############################
#######################################################################


def aucs_to_objective(aucs:np.ndarray): #shape (M, 2, 2)
    """Takes AUCs from 'run_single_kernel_single_label' and outputs the
    objective for Cross Validation for both Conf and Mahal."""
    aucs = np.sum(aucs, axis=2) # sum of ROC AUC and PR AUC
    return aucs #shape (M, 2)



def eval_1_paramdict_1_fold(fold:tuple,
                            class_to_test:Any,
                            param_dict:Dict[str, Any],
                            min_fold_size:int,
                            n_jobs_gram:int=1,
                            verbose:bool=False,
                            ):
    """Evaluates a single fold for a single hyperparameter configuration.

    Args:
        fold (_type_): _description_
        class_to_test (_type_): _description_
        param_dict (_type_): _description_
        min_fold_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    X_train, y_train, X_val, y_val = fold
    SVD_threshold = 10e-10
    SVD_max_rank = min(min_fold_size, 30)
    return_all_levels = True

    # Simple case for most methods
    if "truncated sig" not in param_dict["kernel_name"]:
        # aucs shape (M, 2, 2), M <= min_fold_size
        aucs = np.zeros( (min_fold_size, 2) )
        raw_aucs = run_single_kernel_single_label(X_train, y_train, X_val, y_val,
                            class_to_test, param_dict,
                            SVD_threshold,
                            SVD_max_rank, verbose,
                            return_all_levels=return_all_levels,
                            n_jobs=n_jobs_gram,
                            )
        aucs[:len(raw_aucs)] = aucs_to_objective(raw_aucs)
    # Computationally efficient truncated signature case
    else: 
        MAX_ORDER = 10
        param_dict["order"] = MAX_ORDER

        # Obtain grams once instead of MAX_ORDER times
        corpus, test = get_corpus_and_test(X_train, y_train, X_val, 
                                    class_to_test)
        vv_grams, uv_grams = calc_grams(corpus, test, param_dict, 
                                        sig_kernel_only_last=False, n_jobs=n_jobs_gram,
                                        verbose=verbose)
        
        # Store aucs for each truncation level
        aucs = np.zeros((min_fold_size, 2, MAX_ORDER))
        for idx, (vv, uv) in enumerate(zip(vv_grams, uv_grams)):

            raw_aucs = run_single_kernel_single_label(X_train, 
                            y_train, X_val, y_val,
                            class_to_test, param_dict,
                            SVD_threshold,
                            SVD_max_rank, verbose, vv, uv,
                            return_all_levels=return_all_levels,
                            n_jobs=n_jobs_gram,)
            aucs[:len(raw_aucs), :, idx] = aucs_to_objective(raw_aucs)

    return aucs #auc shape (min_fold_size, 2) or (min_fold_size, 2, n_truncs) for truncated sig



def eval_repeats_folds(kernel_name:str,
                repeats_and_folds:List[List[tuple]],
                hyperparams:List[Dict[str, Any]],  #List of {name : hyperparam_value}
                class_to_test,
                n_jobs_repeats:int = 1,
                n_jobs_gram:int = 1,
                verbose:bool = False,
                ):
    """"We permform anomaly detection using 'class_to_test' as the normal class. 
    We then calculate the AUC scores for the given hyperparameters."""

    #calc minimum size of the label class
    min_fold_size = min([len(np.where(y_train==class_to_test)[0]) 
                            for (_, y_train, _, _) in repeats_and_folds[0]])

    #for each parameter:
    scores = []
    for param_dict in hyperparams:
        param_dict["kernel_name"] = kernel_name
        param_dict["normal_class_label"] = class_to_test

        #loop over repeats and folds
        repeat_scores = Parallel(n_jobs=n_jobs_repeats)(
            delayed(eval_1_paramdict_1_fold)(fold, class_to_test, param_dict, 
                            min_fold_size, n_jobs_gram, verbose)
            for repeats in repeats_and_folds
            for fold in repeats
        )
        #make shape consistent.
        scores.append(repeat_scores)
    
    #average across repeats and folds
    scores = np.array(scores)
    scores = np.mean(scores, axis=(1))
    return scores #shape (n_hyperparams, min_fold_size, 2, (opt. dim: n_truncs))



def choose_best_hyperparam(scores_conf_mahal:np.ndarray,
                           hyperparams:List[Dict[str, Any]],
                        ):
    """Chooses the best hyperparameter configuration based on the AUC 
    scores outputed from 'eval_repeats_folds'."""
    c_m_param_dicts = [{}, {}]
    for i in range(2):
        #scores_conf_mahal shape (n_hyperparams, min_fold_size, 2, (opt. dim: n_truncs))
        scores = scores_conf_mahal[:,:,i]
        dims = np.arange(scores.ndim) 
        max_params = np.max(scores, axis=tuple(dims[1:]) )
        best_param_idx = np.argmax(max_params)
        max_thresh = np.max(scores, axis=(0, *dims[2:]))
        best_thresh_idx = np.argmax(max_thresh)

        #choose best param_dict
        final_param_dict = hyperparams[best_param_idx].copy()
        final_param_dict["threshold"] = 1 + best_thresh_idx
        final_param_dict["CV_train_score"] = max_params[best_param_idx]
        kernel_name = final_param_dict["kernel_name"]

        #store some extra stats
        final_param_dict["score_params"] = max_params
        final_param_dict["score_thresh"] = max_thresh

        #optional: best truncation level
        if "truncated sig" in kernel_name:
            max_truncs = np.max(scores, axis=(0, 1))
            best_trunc_idx = np.argmax(max_truncs)
            final_param_dict["order"] = 1+best_trunc_idx
            final_param_dict["score_orders"] = max_truncs
        c_m_param_dicts[i] = final_param_dict

    return c_m_param_dicts



def cv_given_dataset(X:List,                #Training Dataset
                    y:np.array,             #Training class labels
                    unique_labels:np.array, #Unique class labels
                    kernel_names:List[str],
                    k:int = 4,                  #k-fold cross validation
                    n_repeats:int = 10,         #repeats of k-fold CV
                    n_jobs_repeats:int = 1,
                    n_jobs_gram:int = 1,
                    verbose:bool = False
                    ):
    """Performs repeated k-fold cross-validation on the given dataset 
    for the anomaly detection models specified by 'kernel_names'. We 
    use the AUC scores to evaluate the performance of the models. Saves
    the result as a nested dictionary of the form {kernel : label : params}"""

    repeats_and_folds = repeat_k_folds(X, y, k, n_repeats)

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
            scores = eval_repeats_folds(kernel_name, repeats_and_folds,
                                        hyperparams, label,
                                        n_jobs_repeats, n_jobs_gram, verbose)
            c_param_dict, m_param_dict = choose_best_hyperparam(scores, hyperparams)
            c_labelwise_param_dicts[label] = c_param_dict
            m_labelwise_param_dicts[label] = m_param_dict
        c_kernelwise_param_dicts[kernel_name] = c_labelwise_param_dicts
        m_kernelwise_param_dicts[kernel_name] = m_labelwise_param_dicts

        t1 = time.time()
        print(f"Time taken for kernel {kernel_name}:", t1-t0, "seconds")
    return c_kernelwise_param_dicts, m_kernelwise_param_dicts


def cv_tslearn(dataset_names:List[str], 
                kernel_names:List[str],
                k:int = 5,                  # k-fold cross validation
                n_repeats:int = 10,         # repeats of k-fold CV)
                n_jobs_repeats:int = 1,
                n_jobs_gram:int = 1,
                verbose:bool = False
                ):    
    """Cross validation for tslearn datasets"""
    cv_best_models = {} # dataset_name : kernel_name : label : param_dict
    for dataset_name in dataset_names:
        print("Dataset:", dataset_name)
        # Load dataset
        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)
        N_train, T, d = X_train.shape
        print_dataset_stats(num_classes, d, T, N_train, "N/A")

        # Run each kernel
        t0 = time.time()
        c_kernelwise_param_dicts, m_kernelwise_param_dicts = cv_given_dataset(
                                                X_train, y_train, unique_labels, 
                                                kernel_names, k, n_repeats,
                                                n_jobs_repeats, n_jobs_gram, verbose)
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
    given a dict of form {dataset_name : kernel_name : label : param_dict}"""
    print("Cross Validation Results")
    with np.printoptions(precision=3, suppress=True):
        for dataset_name, results in dataset_kernel_label_paramdict.items():
            print_dataset_stats(results['num_classes'], results['path dim'], 
                                results['ts_length'], results['N_train'], "N/A")
            for anomaly_method in ["conf_results", "mahal_results"]:
                print(f"\n{anomaly_method}")
                kernelwise_dict = results[anomaly_method]
                for kernel_name, labelwise_dict in kernelwise_dict.items():
                    final_score_avgs = average_labels(labelwise_dict, "CV_train_score")
                    params_score_avgs = average_labels(labelwise_dict, "score_params")
                    thresh_score_avgs = average_labels(labelwise_dict, "score_thresh")
                    print(f"\n{kernel_name}")
                    print("final_score_avgs", final_score_avgs)
                    print("params_score_avgs", params_score_avgs)
                    print("thresh_score_avgs", thresh_score_avgs)
                    if "truncated sig" in kernel_name:
                        trunc_score_avgs = average_labels(labelwise_dict, "score_orders")
                        print("orders_score_avgs", trunc_score_avgs)
                    
                    for label, param_dict in labelwise_dict.items():
                        print(label)
                        print({k:v for k,v in param_dict.items() 
                            if k not in ["kernel_name", "normal_class_label", 
                                            "CV_train_score", "score_params", "score_thresh", 
                                            "score_orders"]})
            print("\nEnd dataset \n\n\n")
            


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run this script to run cross validation on ts-learn datasets.")
    parser.add_argument("--dataset_names", nargs="+", type=str, default=[
        'Epilepsy',                    # N_corpus = 34      #I should probably further limit this to 100 < N_corpus < 1000
        'EthanolConcentration',        # N_corpus = 65
        'FingerMovements',             # N_corpus = 158
        'HandMovementDirection',       # N_corpus = 40
        'Heartbeat',                   # N_corpus = 102
        'LSST',                        # N_corpus = 176
        'MotorImagery',                # N_corpus = 139
        'NATOPS',                      # N_corpus = 30
        'PenDigits',                   # N_corpus = 749
        'PEMS-SF',                     # N_corpus = 38
        'PhonemeSpectra',              # N_corpus = 85
        'RacketSports',                # N_corpus = 38
        'SelfRegulationSCP1',          # N_corpus = 134
        ])
    parser.add_argument("--kernel_names", nargs="+", type=str, default=[
                "linear",
                "rbf",
                "poly",
                "integral rbf",
                "integral poly",
                "truncated sig",
                "truncated sig rbf",
                "signature pde rbf",
                "gak",
                ])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--n_jobs_repeats", type=int, default=5)
    parser.add_argument("--n_jobs_gram", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=f"Data/cv_{int(time.time()*1000)}.pkl")
    args = vars(parser.parse_args())
    print("Args:", args)

    cv_best_models = cv_tslearn(
            dataset_names = args["dataset_names"],
            kernel_names = args["kernel_names"],
            k = args["k"],
            n_repeats = args["n_repeats"],
            n_jobs_repeats = args["n_jobs_repeats"],
            n_jobs_gram = args["n_jobs_gram"],
                )
    
    #save to disk
    save_to_pickle(cv_best_models, args["save_path"])
    print_cv_results(cv_best_models)