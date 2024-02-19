import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
from joblib import Memory, Parallel, delayed
import tslearn
import tslearn.metrics
from tslearn.datasets import UCR_UEA_datasets
import pickle
import time

from experiments.cross_validation import cv_given_dataset


def cv_tslearn(dataset_names:List[str], 
                kernel_names:List[str],
                k:int = 5,              #k-fold cross validation
                n_repeats:int = 10,      #repeats of k-fold CV)
                n_jobs_repeats:int = 1,
                n_jobs_gram:int = 1,
                verbose:bool = False
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
                                                n_jobs_repeats, n_jobs_gram, verbose)
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