import numpy as np
import sklearn.metrics
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
from joblib import Memory, Parallel, delayed
import tslearn
import tslearn.metrics
from tslearn.datasets import UCR_UEA_datasets
import pickle
import time

from run_experiments import run_single_kernel_single_label, get_corpus_and_test, calc_grams


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


def eval_1_paramdict_1_fold(fold,
                            class_to_test,
                            param_dict,
                            fixed_length,
                            min_fold_size,
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
        aucs = run_single_kernel_single_label(X_train, y_train, X_val, y_val,
                            class_to_test, param_dict,
                            fixed_length, verbose=False,
                            return_all_levels=True,
                            SVD_threshold=0,
                            SVD_max_rank=min_fold_size,
                            )
        return aucs_to_objective(aucs)

    else: # Computationally efficient truncated signature case
        MAX_ORDER = 15
        param_dict["order"] = MAX_ORDER

        # Obtain grams once instead of MAX_ORDER times
        corpus, test = get_corpus_and_test(X_train, y_train, X_val, 
                    class_to_test, fixed_length)
        vv_grams, uv_grams = calc_grams(corpus, test, param_dict, fixed_length, 
                                        sig_kernel_only_last=False)
        
        # Store aucs for each truncation level
        aucs = np.zeros((MAX_ORDER, min_fold_size))
        for idx, (vv, uv) in enumerate(zip(vv_grams, uv_grams)):

            raw_aucs = run_single_kernel_single_label(X_train, 
                                y_train, X_val, y_val,
                                class_to_test, param_dict,
                                fixed_length, verbose=False,
                                return_all_levels=True,
                                SVD_threshold=0,
                                SVD_max_rank=min_fold_size,
                                vv_gram=vv, uv_gram=uv,)
            
            aucs[idx, :len(raw_aucs)] = aucs_to_objective(raw_aucs)
        return aucs


def eval_repeats_folds(kernel_name:str,
                repeats_and_folds:List[List[tuple]],
                hyperparams:List[Dict[str, Any]],  #List of {name : hyperparam_value}
                class_to_test,
                fixed_length:bool,
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

        #loop over repeats and folds
        repeat_scores = []
        for folds in repeats_and_folds:
            folds_scores = []
            for fold in folds:
                #auc shape (M,) or (n_truncs, M) for truncated sig
                auc = eval_1_paramdict_1_fold(fold, class_to_test,
                            param_dict, fixed_length, min_fold_size)
                fix_shape = (*auc.shape[:-1], min_fold_size)
                score = np.zeros(fix_shape)
                M = auc.shape[-1]
                score[..., :M] = auc[..., :M]

                folds_scores.append(score)
            repeat_scores.append(folds_scores)
        scores.append(repeat_scores)
    
    #average across repeats and folds
    scores = np.array(scores)
    scores = np.mean(scores, axis=(1, 2))
    return scores #shape (n_hyperparams, min_fold_size, (opt. dim: n_truncs))


def cv_given_dataset(X:List,                #Training Dataset
                    y:np.array,             #Training class labels
                    unique_labels:np.array, #Unique class labels
                    kernel_names:List[str],
                    fixed_length:bool,
                    k:int = 4,          #k-fold cross validation
                    n_repeats:int = 2,  #repeats of k-fold CV
                    ):
    """Performs repeated k-fold cross-validation on the given dataset 
    for the anomaly detection models specified in 'kernel_names'. We 
    use the AUC scores to evaluate the performance of the models.
    TODO saves the models as..... list of dict..... idk"""

    repeats_and_folds = repeat_k_folds(X, y, k, n_repeats)

    kernelwise_param_dicts = {} # kernel : label : param_dict
    for kernel_name in kernel_names:
        hyperparams = get_hyperparam_combinations(kernel_name)

        #loop over normal class
        labelwise_param_dicts = {} # label : param_dict
        t0 = time.time()
        for label in tqdm(unique_labels, 
                          desc=f"Label for {kernel_name}"):
    
            #scores shape (n_hyperparams, min_fold_size, (opt. dim: n_truncs))
            scores = eval_repeats_folds(kernel_name, repeats_and_folds,
                                        hyperparams, label, fixed_length)

            dims = np.arange(scores.ndim) 

            max_params = np.max(scores, axis=tuple(dims[1:]) )
            best_param_idx = np.argmax(max_params)
            max_thresh = np.max(scores, axis=(0, *dims[2:]))
            best_thresh_idx = np.argmax(max_thresh)

            final_param_dict = hyperparams[best_param_idx].copy()
            final_param_dict["threshold"] = 1 + best_thresh_idx
            final_param_dict["normal_class_label"] = label
            final_param_dict["CV_train_auc"] = max_params[best_param_idx]

            #optional: best truncation level
            if "truncated sig" in kernel_name:
                max_truncs = np.max(scores, axis=(0, 1))
                best_trunc_idx = np.argmax(max_truncs)
                final_param_dict["order"] = 1+best_trunc_idx
            
            #store
            labelwise_param_dicts[label] = final_param_dict
        kernelwise_param_dicts[kernel_name] = labelwise_param_dicts
        t1 = time.time()
        print(f"Time taken for kernel {kernel_name}:", t1-t0, "seconds")
    return kernelwise_param_dicts


def cv_tslearn(dataset_names:List[str], 
                kernel_names:List[str],
                k:int = 5,              #k-fold cross validation
                n_repeats:int = 10,      #repeats of k-fold CV)
                ):    
    """Cross validation for tslearn datasets"""

    cv_best_models = {} # dataset_name : kernel_name : label : param_dict
    for dataset_name in dataset_names:
        print("Dataset:", dataset_name)
        # Load dataset
        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

        # stats
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)
        N_train, T, d = X_train.shape

        # Run each kernel
        t0 = time.time()
        kernelwise_param_dicts = cv_given_dataset(X_train, y_train, unique_labels, 
                                                kernel_names, fixed_length=True,
                                                k=k, n_repeats=n_repeats)
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
    with open("CV_tslearn.pkl", 'wb') as handle:
        pickle.dump(cv_best_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cv_best_models



#NEXT: TODO TODO TODO either 
#   1) ---DONE--- add truncated signatures support
#or 2) ---DONE--- create code that saves the best result for each dataset, kernel, label
#or 3) ---DONE--- add for loop for all datasets
#or 4) make the evaluation on test set work with param_dict
#or 5) add joblib integration

if __name__ == "__main__":
    cv_best_models = cv_tslearn(
            dataset_names = [
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
                ],
            kernel_names = [
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
                ],
                )