import pickle
from typing import List, Optional, Dict, Set, Callable, Any


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