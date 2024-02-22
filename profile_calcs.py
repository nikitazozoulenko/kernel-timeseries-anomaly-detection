from memory_profiler import profile
import numpy as np
from experiments.experiment_code import calc_grams


def profile_calcs(N=10, T=150, d=60, order=10):
    X = np.random.rand(N, T, d)
    param_dict = {"kernel_name": "truncated sig", 
                  "order": order}
    grams = calc_grams(X, X, param_dict, fixed_length=True)
    return grams

if __name__ == "__main__":
    profile_calcs()