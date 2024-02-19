import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
from joblib import Memory, Parallel, delayed
import pandas as pd
import pickle
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiment_code import print_dataset_stats
from cross_validation import cv_given_dataset
from models.signature import transform_stream


#######################################################################################################################
## Original loading code taken from https://github.com/pafoster/conformance_distance_experiments_cochrane_et_al_2020 ##
## DATASET_URLS = ['https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits-orig.tes.Z',       ##
##                 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits-orig.tra.Z']       ##
#######################################################################################################################


def read_pendigits_file(filename):
    with open(filename, 'r') as f:
        data_lines = f.readlines()

    data = []
    data_labels = []
    current_digit = None
    for line in data_lines:
        if line == "\n":
            continue

        if line[0] == ".":
            if "SEGMENT DIGIT" in line[1:]:
                if current_digit is not None:
                    data.append(np.array(current_digit))
                    data_labels.append(digit_label)

                current_digit = []
                digit_label = int(line.split('"')[1])
            else:
                continue

        else:
            x, y = map(float, line.split())
            current_digit.append([x, y])
            
    data.append(np.array(current_digit))
    data_labels.append(digit_label)
    return data, np.array(data_labels)


def create_pendigits_df():
    data = {'train': read_pendigits_file("Data/pendigits-orig.tra"),
            'test': read_pendigits_file("Data/pendigits-orig.tes")}

    #turn to df
    dataframes = []
    for subset, data in data.items():
        df = pd.DataFrame(data).T
        df.columns = ['data', 'label']
        df['subset'] = subset
        dataframes.append(df)
    df = pd.concat(dataframes)

    #apply stream transforms
    transforms = lambda s : transform_stream(s, stream_transforms=["min_max_normalize"])
    df["data"] = df["data"].apply(transforms)
    return df


#############################################################
##########        Cross Validation code           ###########
#############################################################


def cv_pendigits(
                kernel_names:List[str],
                k:int = 5,              #k-fold cross validation
                n_repeats:int = 10,      #repeats of k-fold CV)
                n_jobs_repeats:int = 1,
                n_jobs_gram:int = 1,
                verbose:bool = False
                ):    
    """Cross Validation for PenDigits dataset. df has columns 
    ["data", "label", "subset"]. Each data point is a timeseries 
    of shape (T_i, d) of variable length."""
    current_time = int(time.time())

    #Gather dataset info
    df = create_pendigits_df()
    X_train = df[df["subset"]=="train"]["data"].values
    y_train = np.array(df[df["subset"]=="train"]["label"].values)
    X_test = df[df["subset"]=="test"]["data"].values
    y_test = np.array(df[df["subset"]=="test"]["label"].values)
    unique_labels = sorted(df["label"].unique())
    num_classes = len(unique_labels)
    d = X_train[0].shape[1]
    T = "variable length"
    N_train = len(X_train)
    N_test = len(X_test)
    print_dataset_stats(num_classes, d, T, N_train, N_test)

    # Run each kernel
    t0 = time.time()
    kernelwise_param_dicts = cv_given_dataset(X_train, y_train, unique_labels, 
                                            kernel_names, True, k, n_repeats, #fixed_length = True
                                            n_jobs_repeats, n_jobs_gram, verbose)
    t1 = time.time()
    print(f"Time taken for PenDigits:", t1-t0, "seconds\n\n\n")
        
    #log dataset experiment
    cv_best_models = {}
    cv_best_models["PenDigits"] = {"kernel_results": kernelwise_param_dicts, 
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
    parser = argparse.ArgumentParser(description="Run this script to run cross validation on variable length PenDigits.")
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

    cv_best_models = cv_pendigits(
            dataset_names = args["dataset_names"],
            kernel_names = args["kernel_names"],
            k = args["k"],
            n_repeats = args["n_repeats"],
            n_jobs_repeats = args["n_jobs_repeats"],
            n_jobs_gram = args["n_jobs_gram"],
                )
    
    print(cv_best_models)