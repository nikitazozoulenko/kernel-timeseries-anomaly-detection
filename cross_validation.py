import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
from joblib import Memory, Parallel, delayed
import tslearn
import tslearn.metrics
from tslearn.datasets import UCR_UEA_datasets
import pickle
import time

from experiment_code import run_single_kernel_single_label, get_corpus_and_test, calc_grams


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
    n_sigmas = 6
    ranges = {}

    #static kernel params. Note that sig and integral kernels also use this
    if "rbf" in kernel_name:
        ranges["sigma"] = np.exp(np.linspace(-5, 0, n_sigmas))
    elif "poly" in kernel_name:
        ranges["p"] = np.arange(2, max_poly_p+1)

    return ranges
    
#######################################################################
######################### Cross Validation ############################
#######################################################################


def aucs_to_objective(aucs:np.ndarray): #shape (M, 2, 2)
    """Takes AUCs from 'run_single_kernel_single_label' and outputs the
    objective for Cross Validation."""
    aucs = np.max(aucs, axis=1) # max of conf and mahal
    aucs = aucs[:, 0]           # only interested in roc (and not pr)
    return aucs #shape (M,)



def eval_1_paramdict_1_fold(fold:tuple,
                            class_to_test:Any,
                            param_dict:Dict[str, Any],
                            fixed_length:bool,
                            min_fold_size:int,
                            n_jobs_gram:int=1,
                            ):
    """Evaluates a single fold for a single hyperparameter configuration.

    Args:
        fold (_type_): _description_
        class_to_test (_type_): _description_
        param_dict (_type_): _description_
        fixed_length (_type_): _description_
        min_fold_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    X_train, y_train, X_val, y_val = fold

    # Simple case for most methods
    if "truncated sig" not in param_dict["kernel_name"]:
        # aucs shape (M, 2, 2), M <= min_fold_size
        aucs = np.zeros(min_fold_size)
        raw_aucs = run_single_kernel_single_label(X_train, y_train, X_val, y_val,
                            class_to_test, param_dict,
                            fixed_length, verbose=False,
                            return_all_levels=True,
                            SVD_threshold=0,
                            SVD_max_rank=min_fold_size,
                            n_jobs=n_jobs_gram,
                            )
        aucs[:len(raw_aucs)] = aucs_to_objective(raw_aucs)
    # Computationally efficient truncated signature case
    else: 
        MAX_ORDER = 15
        param_dict["order"] = MAX_ORDER

        # Obtain grams once instead of MAX_ORDER times
        corpus, test = get_corpus_and_test(X_train, y_train, X_val, 
                    class_to_test, fixed_length)
        vv_grams, uv_grams = calc_grams(corpus, test, param_dict, fixed_length, 
                                        sig_kernel_only_last=False, n_jobs=n_jobs_gram)
        
        # Store aucs for each truncation level
        aucs = np.zeros((min_fold_size, MAX_ORDER))
        for idx, (vv, uv) in enumerate(zip(vv_grams, uv_grams)):

            raw_aucs = run_single_kernel_single_label(X_train, 
                                y_train, X_val, y_val,
                                class_to_test, param_dict,
                                fixed_length, verbose=False,
                                return_all_levels=True,
                                SVD_threshold=0,
                                SVD_max_rank=min_fold_size,
                                vv_gram=vv, uv_gram=uv, n_jobs=n_jobs_gram)
            aucs[idx, :len(raw_aucs)] = aucs_to_objective(raw_aucs)

    return aucs #auc shape (min_fold_size,) or (min_fold_size, n_truncs) for truncated sig



def eval_repeats_folds(kernel_name:str,
                repeats_and_folds:List[List[tuple]],
                hyperparams:List[Dict[str, Any]],  #List of {name : hyperparam_value}
                class_to_test,
                fixed_length:bool,
                n_jobs_repeats:int = 1,
                n_jobs_gram:int = 1,
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
            delayed(eval_1_paramdict_1_fold)(fold, class_to_test,
                        param_dict, fixed_length, min_fold_size, n_jobs_gram)
            for repeats in repeats_and_folds
            for fold in repeats
        )
        scores.append(repeat_scores)
    
    #average across repeats and folds
    scores = np.array(scores)
    scores = np.mean(scores, axis=(1))
    return scores #shape (n_hyperparams, min_fold_size, (opt. dim: n_truncs))



def choose_best_hyperparam(scores:np.ndarray,
                           hyperparams:List[Dict[str, Any]],
                        ):
    """Chooses the best hyperparameter configuration based on the AUC 
    scores outputed from 'eval_repeats_folds'."""
    
    #scores shape (n_hyperparams, min_fold_size, (opt. dim: n_truncs))
    dims = np.arange(scores.ndim) 
    max_params = np.max(scores, axis=tuple(dims[1:]) )
    best_param_idx = np.argmax(max_params)
    max_thresh = np.max(scores, axis=(0, *dims[2:]))
    best_thresh_idx = np.argmax(max_thresh)

    #choose best param_dict
    final_param_dict = hyperparams[best_param_idx].copy()
    final_param_dict["threshold"] = 1 + best_thresh_idx
    final_param_dict["CV_train_auc"] = max_params[best_param_idx]
    kernel_name = final_param_dict["kernel_name"]

    #store some extra stats
    final_param_dict["auc_params"] = max_params
    final_param_dict["auc_thresh"] = max_thresh

    #optional: best truncation level
    if "truncated sig" in kernel_name:
        max_truncs = np.max(scores, axis=(0, 1))
        best_trunc_idx = np.argmax(max_truncs)
        final_param_dict["order"] = 1+best_trunc_idx
        final_param_dict["auc_orders"] = max_truncs
    
    return final_param_dict



def cv_given_dataset(X:List,                #Training Dataset
                    y:np.array,             #Training class labels
                    unique_labels:np.array, #Unique class labels
                    kernel_names:List[str],
                    fixed_length:bool,
                    k:int = 4,                  #k-fold cross validation
                    n_repeats:int = 10,         #repeats of k-fold CV
                    n_jobs_repeats:int = 1,
                    n_jobs_gram:int = 1,
                    ):
    """Performs repeated k-fold cross-validation on the given dataset 
    for the anomaly detection models specified by 'kernel_names'. We 
    use the AUC scores to evaluate the performance of the models. Saves
    the result as a nested dictionary of the form {kernel : label : params}"""

    repeats_and_folds = repeat_k_folds(X, y, k, n_repeats)

    kernelwise_param_dicts = {} # kernel : label : param_dict
    for kernel_name in kernel_names:
        hyperparams = get_hyperparam_combinations(kernel_name)

        #loop over normal class
        labelwise_param_dicts = {} # label : param_dict
        t0 = time.time()
        for label in tqdm(unique_labels, desc = f"Label for {kernel_name}"):
            scores = eval_repeats_folds(kernel_name, repeats_and_folds,
                                        hyperparams, label, fixed_length,
                                        n_jobs_repeats, n_jobs_gram)
            final_param_dict = choose_best_hyperparam(scores, hyperparams)
            labelwise_param_dicts[label] = final_param_dict
        kernelwise_param_dicts[kernel_name] = labelwise_param_dicts

        t1 = time.time()
        print(f"Time taken for kernel {kernel_name}:", t1-t0, "seconds")
    return kernelwise_param_dicts



def cv_tslearn(dataset_names:List[str], 
                kernel_names:List[str],
                k:int = 5,              #k-fold cross validation
                n_repeats:int = 10,      #repeats of k-fold CV)
                n_jobs_repeats:int = 1,
                n_jobs_gram:int = 1,
                ):    
    """Cross validation for tslearn datasets"""
    current_time = int(time.time())

    cv_best_models = {} # dataset_name : kernel_name : label : param_dict
    for dataset_name in dataset_names:
        print("Dataset:", dataset_name)
        # Load dataset
        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)
        N_train, T, d = X_train.shape

        # Run each kernel
        t0 = time.time()
        kernelwise_param_dicts = cv_given_dataset(X_train, y_train, unique_labels, 
                                                kernel_names, True, k, n_repeats, #fixed_length = True
                                                n_jobs_repeats, n_jobs_gram)
        t1 = time.time()
        print(f"Time taken for dataset {dataset_name}:", t1-t0, "seconds\n\n\n")
        
        #log dataset experiment
        cv_best_models[dataset_name] = {"kernel_results": kernelwise_param_dicts, 
                                     "num_classes": num_classes, 
                                     "path dim":d,
                                     "ts_length":T, 
                                     "N_train":N_train
                                     }
    #save to disk
    with open(f"CV_tslearn_{current_time}.pkl", 'wb') as handle:
        pickle.dump(cv_best_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cv_best_models



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run this script to run cross validation on ts-learn datasets.")
    parser.add_argument("--dataset_names", nargs="+", type=str, default=[
            #'ArticularyWordRecognition', 
            #'BasicMotions', 
            #'Cricket',
            ##########'ERing', #cant find dataset
            'Libras', 
            #'NATOPS', 
            #'RacketSports',     
            #'FingerMovements',
            #'Heartbeat',
            #'SelfRegulationSCP1', 
            #'UWaveGestureLibrary'
        ])
    parser.add_argument("--kernel_names", nargs="+", type=str, default=[
                "linear",
                "rbf",
                "poly",
                "gak",
                "truncated sig",
                "truncated sig rbf",
                "truncated sig poly",
                "signature pde",
                "signature pde rbf",
                "signature pde poly",
                "integral linear",
                "integral rbf",
                "integral poly",
                ])

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--n_jobs_repeats", type=int, default=50)
    parser.add_argument("--n_jobs_gram", type=int, default=1)

    args = vars(parser.parse_args())
    print(args)

    cv_best_models = cv_tslearn(
            dataset_names = args["dataset_names"],
            kernel_names = args["kernel_names"],
            k = args["k"],
            n_repeats = args["n_repeats"],
            n_jobs_repeats = args["n_jobs_repeats"],
            n_jobs_gram = args["n_jobs_gram"],
                )
    
    print(cv_best_models)