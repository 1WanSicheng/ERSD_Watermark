import numpy as np
import torch
import tqdm
import pickle
import argparse
from .gumbel_detect_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--save_path', type=str, help='Save path')
    args = parser.parse_args()
    print(args)

    with open(args.data_path, 'rb') as f:
        wm_data = pickle.load(f)

    accepts_len = [np.mean(wm_data[i]['gen_seq_lens']) for i in range(len(wm_data))]
    print(f"Average accept length: {np.mean(accepts_len)}")

    Y, Y_mc = get_pivotals(wm_data, 200)
    Y_comb = (Y+Y_mc)/2
    R = get_r_values(wm_data, 200)
    assert R.shape == Y.shape == Y_mc.shape

    p = (np.mean(accepts_len) - 1)/np.mean(accepts_len)
    r_random_list = []
    for _ in range(10):
        r_random, _ = ars_with_random_mix(Y, Y_mc, p=p)
        r_random_list.append(r_random)
    
    res = run_repeated_threshold_training(
        Y, Y_mc, R,
        trials=10, ratio=1.0, alpha=0.01,
        objective="final"
    )

    results = {'Ars-tau': res, 'Ars-prior': r_random_list}
    with open(args.save_path, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()