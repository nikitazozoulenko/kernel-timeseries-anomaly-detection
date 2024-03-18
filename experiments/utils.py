import pickle
from typing import List, Optional, Dict, Set, Callable, Any, Literal
import numpy as np


##########################################################################
######################## Pickle save and load ############################
##########################################################################


def join_dicts_from_pickle_paths(paths:List[str]) -> Dict:
    """Reads a list of pickles and joins them into a single dict.

    Args:
        paths (List[str]): List of paths to pickles.

    Returns:
        Dict: Joined dicts.
    """
    dicts = [load_from_pickle(path)
             for path in paths]
    joined_dicts = {}
    for d in dicts:
        joined_dicts.update(d)
    return joined_dicts


def save_to_pickle(obj:Any, 
                   path:str = "Data/saved.pkl"
                   ) -> None:
    """Saves an object to a pickle file.

    Args:
        obj (Any): Object to be saved.
        path (str, optional): Path to save the pickle. 
                              Defaults to "Data/saved.pkl".

    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(path:str = "Data/saved.pkl") -> Any:
    """Loads an object from a pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


##########################################################################
################ Parse test results into a LaTeX table ###################
##########################################################################


def highlight_best(scores:np.ndarray[str],
                   max_or_min:Literal["max", "min"],
                   ):
    """" Given a ndim=1 array of scores as strings,
    replace all best occurences with a boldface string."""
    best = max(scores) if max_or_min == "max" else min(scores)
    best = np.argwhere(np.array(scores) == best).flatten()
    copy = [s for s in scores]
    for i in best:
        copy[i] = r"\textbf{" + scores[i] + "}"
    return copy



def retrieve_kernel_AUCs(kernelwise_dict:Dict[str, np.ndarray],
                  mahal_or_conf:Literal["conf", "mahal"],
                  order=["flat linear", "flat rbf", "flat poly", "integral rbf", "integral poly", "trunc sig linear", "trunc sig rbf", "pde sig rbf", "gak", "reservoir"],
                  ):
    """Retrieves the AUCs for each kernel in the order given by the list `order`."""
    #reorder the kernels
    assert set(kernelwise_dict.keys()) == set(order)
    scores = [kernelwise_dict[k] for k in order]

    ROC_AUC = []
    PR_AUC = []
    for scores in scores:
        if mahal_or_conf == "conf":
            ROC_AUC.append(scores[0, 0])
            PR_AUC.append( scores[0, 1])
        elif mahal_or_conf == "mahal":
            ROC_AUC.append( scores[1, 0])
            PR_AUC.append( scores[1, 1])
        else:
            raise ValueError("Argument mahal_or_conf must be 'conf' or 'mahal'")
    return np.array(ROC_AUC), np.array(PR_AUC) #ndim=1



def add_datasets_to_latex_table(
    arr:np.ndarray, #shape (n_datasets, 2, n_kernels), axis=1 is [conf, mahal]
    column_names:List[str],
    round_digits:int, 
    leading_zero:bool,
    max_or_min:Literal["max", "min"] = "max" #used for boldfacing the best results
):
    """Given dataset and kernel results, return a string of LaTeX code 
    for the correspondig rows in a table.

    Args:
        arr (np.ndarray): Array with shape (n_datasets, 2, n_kernels) 
                          containing the AUCs.
        column_names (List[str]): Name of columns, e.g. dataset names.
        round_digits (int): Number of digits to round to.
        leading_zero (bool): Whether to keep leading zeros.
    """
    code = ""
    for dataset_name, cm_ker_scores in zip(column_names, arr):
        code += "\t\t\\hline\n\t\t"
        code += r"\multirow{2}{*}{" + dataset_name + r"}    " + "\n"
        for cm, ker_scores in zip(["C", "M"], cm_ker_scores):
            results = [f"{score:.{round_digits}f}" for score in ker_scores]
            results = [s.lstrip('0') if not leading_zero else s for s in results]
            results = highlight_best(results, max_or_min)
            code += "\t\t& " + cm + " & " + " & ".join(results) + r"\\" + "\n"
    return code



def latex_table(arr:np.ndarray, #shape (n_datasets, 2, n_kernels), axis=1 is [conf, mahal]
          dataset_names:List[str],
          round_digits:int,
          leading_zero:bool,
          title:str):
    """Given dataset and kernel results, return string of LaTeX code 
    for a table

    Args:
        arr (np.ndarray): Array with shape (n_datasets, 2, n_kernels) 
                          containing the AUCs.
        dataset_names (List[str]): List of the dataset names.
        round_digits (int): Number of digits to round to.
        leading_zero (bool): Whether to keep leading zeros.
    """
    #Add start of table
    code = r"""
    \begin{tabular}{lc||ccc|cc|ccc|c|c}
        \toprule
        \multirow{2}{*}{Dataset}   &  \multicolumn{11}{c}{""" + title + r"} \\"
    code += r"""
        \cline{3-12}
                                & & linear & RBF & poly 
                                & $I_\text{RBF}$ & $I_\text{poly}$ 
                                & $S_\text{lin}$ & $S_\text{RBF}$ & $S^\infty_\text{RBF}$ 
                                & GAK & VRK\\ 
        \hline
        \hline""" + "\n"
    
    #Add results for each dataset
    code += add_datasets_to_latex_table(arr, dataset_names, round_digits, leading_zero)
    code += "\t\t\\hline\n\t\t\\hline\n"

    #Add averages
    averages_auc = np.mean(arr, axis=0)
    argsort = np.argsort(1/(arr+1), axis=2)
    averages_rank = np.mean(np.argsort(np.argsort(1/(arr+1), axis=2), axis=2), axis=0) +1
    code += add_datasets_to_latex_table([averages_auc], ["Avg. AUC"], round_digits, leading_zero, "max")
    code += add_datasets_to_latex_table([averages_rank], ["Avg. Rank"], round_digits, leading_zero, "min")

    #Add the bottom
    code += "\t\t" + r"""\bottomrule
    \end{tabular}
    """
    return code


def print_latex_results(experiments:Dict, #given by validate_tslearn
                        round_digits=2, 
                        leading_zero=False):
    """Take the results from validate_tslearn and prints the AUCs in LaTeX format.

    Args:
        experiments (Dict): Dict with the results from validate_tslearn.
        round_digits (int): Number of digits to round to.
        leading_zero (bool): Whether to keep leading zeros.
    """
    #parse dict/json results into score array
    pr_results = []
    roc_results = []
    dataset_names = []
    for dataset_name, results in experiments.items():
        c_roc, c_pr = retrieve_kernel_AUCs(results["conf_results"], "conf")
        m_roc, m_pr = retrieve_kernel_AUCs(results["mahal_results"], "mahal")
        pr_results.append([c_pr, m_pr])
        roc_results.append([c_roc, m_roc])
        dataset_names.append(dataset_name)
    
    #rename datasets
    new_dataset_names = {'Epilepsy':"EP", 
                        'EthanolConcentration':"EC",
                        'FingerMovements':"FM",
                        'HandMovementDirection':"HMD",
                        'Heartbeat':"HB",
                        'MotorImagery':"MI",
                        'NATOPS':"NATO",
                        'PEMS-SF':"PEMS",
                        'RacketSports':"RS", 
                        'SelfRegulationSCP1':"SRS1", 
                        'SelfRegulationSCP2':"SRS2", 
                        'PhonemeSpectra':"PS", 
                        'LSST':"LSST",
                        }
    dataset_names = [new_dataset_names[name] if name in new_dataset_names.keys() else name
                     for name in dataset_names]
    
    #produce the LaTeX tables
    pr_table = latex_table(np.array(pr_results), dataset_names, round_digits, leading_zero, "Precision-Recall AUC")
    auc_table = latex_table(np.array(roc_results), dataset_names, round_digits, leading_zero, "ROC AUC")
    print("PR LaTeX table:")
    print(pr_table)
    print("\n\n\n\n\nROC_AUC LaTeX table:")
    print(auc_table)