import pickle
from typing import List, Optional, Dict, Set, Callable, Any


def save_to_pickle(obj:Any, 
                   path:str = "Data/saved.pkl"):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_from_pickle(path:str = "Data/saved.pkl"):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj