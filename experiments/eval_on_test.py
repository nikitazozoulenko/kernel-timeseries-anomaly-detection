import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Set, Callable, Any
import tslearn
import tslearn.metrics
from tslearn.datasets import UCR_UEA_datasets
import pickle
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.experiment_code import run_all_kernels, print_dataset_stats
from experiments.utils import save_to_pickle, load_from_pickle
from experiments.cross_validation import print_cv_results


def validate_tslearn(
        dataset_kernel_label_paramdict : Dict[str, Dict[str, Dict[str, Any]]],
        n_jobs:int = 1, 
        verbose:bool=False,
        ):
    """Validates the best models from cross validation on the
    tslearn datasets using kernel conformance scores."""
    print("Start validation on test sets")
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
        t0 = time.perf_counter()
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
        t1 = time.perf_counter()
        print(f"Total elapsed time for {dataset_name}: {t1-t0} seconds\n")
    print("End validation on test sets\n\n\n")
    return experiments



def print_test_results(experiments, round_digits=3):
    print("Test Results\n")
    for dataset_name, results in experiments.items():
        #Dataset:
        print("Dataset:", dataset_name)
        print_dataset_stats(results["num_classes"], results["path dim"], 
                            results["ts_length"], results["N_train"], 
                            results["N_test"])

        #Results for each kernel:
        for kernel_name, scores in results["results"].items():
            print("\nKernel:", kernel_name)
            print("Conformance AUC:", round(scores[0, 0], round_digits))
            print("Mahalanobis AUC:", round(scores[1, 0], round_digits))
            print("Conformance PR AUC:", round(scores[0, 1], round_digits))
            print("Mahalanobis PR AUC:", round(scores[1, 1], round_digits))

        print("\nEnd Dataset\n\n\n")



#python3 experiments/eval_on_test.py --cv_datasetwise_dict_paths "Data/cv_results_Heart.pkl" "Data/cv_results_UWave_BM.pkl" --n_jobs_gram 4
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run this script to run cross validation on ts-learn datasets.")
    parser.add_argument("--cv_datasetwise_dict_paths", nargs="+", type=str, default=["Data/cv_results.pkl"])
    parser.add_argument("--n_jobs_gram", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=f"Data/eval_{int(time.time()*1000)}.pkl")
    args = vars(parser.parse_args())
    print("Args:", args)

    # Load the cross validation results
    dicts = [load_from_pickle(path)
             for path in args["cv_datasetwise_dict_paths"]]
    dataset_kernel_label_paramdict = {}
    for d in dicts:
        dataset_kernel_label_paramdict.update(d)
    print_cv_results(dataset_kernel_label_paramdict)

    #run test
    test_results = validate_tslearn(
            dataset_kernel_label_paramdict,
            n_jobs = args["n_jobs_gram"],
                )

    #save test results
    save_to_pickle(test_results, args["save_path"])
    print_test_results(test_results)