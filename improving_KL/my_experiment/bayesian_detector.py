import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import tqdm
import pickle
import gc
import os
import argparse
from .bayesian_detector_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_training', type=str, default='data_root/eli5/train', help='Model name')
    parser.add_argument('--data_path_testing', type=str, default='data_root/eli5/test', help='Model name')
    parser.add_argument('--truncation_length', type=int, default=200, help='Truncation length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-3, help='Learning rate')
    args = parser.parse_args()
    print(args)

    # Load data
    train_path = os.path.join(os.path.dirname(__file__), '..', 'data_root', 'eli5', 'train')
    test_path = os.path.join(os.path.dirname(__file__), '..', 'data_root', 'eli5', 'test')

    train_data = []
    test_data = []
    for file in os.listdir(train_path):
        with open(os.path.join(train_path, file), 'rb') as f:
            data = pickle.load(f)
        train_data.extend(data)
    for file in os.listdir(test_path):
        with open(os.path.join(test_path, file), 'rb') as f:
            data = pickle.load(f)
        test_data.extend(data)

    # Process data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_g_values, train_r_values, train_masks, train_labels, cv_g_values, cv_r_values, cv_masks, cv_labels = process_data_for_training(train_data, args.truncation_length, device)
    test_g_values, test_r_values, test_masks, test_labels = process_data_for_testing(test_data, args.truncation_length, device)

    # Train basic Bayesian detector
    print('='*50)
    print('Training basic Bayesian detector...')
    best_detector_prior = None
    highest_tpr_prior = 0

    for l2_weight in np.logspace(-3, -2, num=4):
        print(f'Training with L2 weight: {l2_weight}')
        detector_module_prior = BayesianDetectorModule(
                        watermarking_depth=30, prior_accept=True).to(device)

        _, max_val_tpr_prior = train_model(
                        detector_module=detector_module_prior,
                        g_values=train_g_values,
                        r_values=train_r_values,
                        mask=train_masks,
                        watermarked=train_labels,
                        g_values_val=cv_g_values,
                        r_values_val=cv_r_values,
                        mask_val=cv_masks,
                        watermarked_val=cv_labels,
                        l2_weight=l2_weight,
                        learning_rate=args.learning_rate,
                        epochs=args.num_epochs,
                        verbose=False,
                    )

        if max_val_tpr_prior > highest_tpr_prior:
            highest_tpr_prior = max_val_tpr_prior
            best_detector_prior = detector_module_prior

    # evaluate on test set
    print(f'Training: Best TPR at FPR 1% for original model: {highest_tpr_prior}')
    pred_prior = batched_predict(best_detector_prior, test_g_values, test_r_values, test_masks, device)
    true_label = test_labels
    tpr_prior = tpr_at_fpr(pred_prior, true_label)
    print(f'Testing: TPR at FPR 1% for original model: {tpr_prior}')

    # Train Bayesian detector with simple accept model
    print('='*50)
    print('Training Bayesian detector with simple accept model...')
    best_detector_sig = None
    highest_tpr = 0

    for l2_weight in np.logspace(-3, -2, num=4):
        print(f'Training with L2 weight: {l2_weight}')
        detector_module_sig = BayesianDetectorModule(
                    watermarking_depth=30).to(device)

        _, max_val_tpr = train_model(
                        detector_module=detector_module_sig,
                        g_values=train_g_values,
                        r_values=train_r_values,
                        mask=train_masks,
                        watermarked=train_labels,
                        g_values_val=cv_g_values,
                        r_values_val=cv_r_values,
                        mask_val=cv_masks,
                        watermarked_val=cv_labels,
                        l2_weight=l2_weight,
                        learning_rate=args.learning_rate,
                        epochs=args.num_epochs,
                        verbose=False,
                    )

        if max_val_tpr > highest_tpr:
            highest_tpr = max_val_tpr
            best_detector_sig = detector_module_sig


    print(f'Training: Best TPR at FPR 1% for r-info model: {highest_tpr}')
    pred = batched_predict(best_detector_sig, test_g_values, test_r_values, test_masks, device)
    true_label = test_labels
    tpr = tpr_at_fpr(pred, true_label)
    print(f'Testing: TPR at FPR 1% for r-info model: {tpr}')

    # Train Bayesian detector with thresholdnet
    print('='*50)
    print('Training Bayesian detector with thresholdnet...')
    best_detector = None
    highest_tpr = 0

    for l2_weight in np.logspace(-3, -2, num=4):
        print(f'Training with L2 weight: {l2_weight}')
        detector_module = BayesianDetectorModule(
                watermarking_depth=30, thresholdnet=True).to(device)

        _, max_val_tpr = train_model(
                        detector_module=detector_module,
                        g_values=train_g_values,
                        r_values=train_r_values,
                        mask=train_masks,
                        watermarked=train_labels,
                        g_values_val=cv_g_values,
                        r_values_val=cv_r_values,
                        mask_val=cv_masks,
                        watermarked_val=cv_labels,
                        l2_weight=l2_weight,
                        learning_rate=args.learning_rate,
                        epochs=args.num_epochs,
                        verbose=True,
                    )

        if max_val_tpr > highest_tpr:
            highest_tpr = max_val_tpr
            best_detector = detector_module

    print(f'Training: Best TPR at FPR 1% for r-info nn model: {highest_tpr}')
    pred = batched_predict(best_detector, test_g_values, test_r_values, test_masks, device)
    true_label = test_labels
    tpr = tpr_at_fpr(pred, true_label)
    print(f'Testing: TPR at FPR 1% for r-info nn model: {tpr}')


if __name__ == '__main__':
    main()