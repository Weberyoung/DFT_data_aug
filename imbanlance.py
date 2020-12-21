import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.approximation import DiscreteFourierTransform
from utils import load_ucr, stratify_by_label
import random



def get_dft_coefs(X, n_coefs):
    """
    :param X: the training set  numpy array shape:[n_samples, n_time_steps]
    :param n_coefs: number of the coefficients
    :return: numpy array shape: [n_samples, n_coefs]
    """
    dft = DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=False,
                                   norm_std=False)
    X_dft = dft.fit_transform(X)
    n_samples = len(X)
    if n_coefs % 2 == 0:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                            np.zeros((n_samples,))]
        ]
    else:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs + 1, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
        ]
    return X_dft_new


def plot_dft(X, X_irfft, n_coefs):
    plt.figure(figsize=(6, 4))
    plt.plot(X, 'o--', ms=4, label='Original')
    plt.plot(X_irfft, 'o--', ms=4, label='DFT - {0} coefs'.format(n_coefs))
    plt.legend(loc='best', fontsize=10)
    plt.xlabel('Time', fontsize=14)
    plt.title('Discrete Fourier Transform', fontsize=16)
    plt.show()


def data_aug_by_dft(stratified_data, n_group):
    """
    :param stratified_data: the data have been stratified by the corresponding label
    :param n_group:
    :return:
    """
    # np.random.seed(66)
    data_augmented = []
    # deal with each label
    # Get the max number of samples of each category
    max_samples = get_max_samples(stratified_data)
    print('The max number of samples: %d' % max_samples)
    for i in range(stratified_data.shape[0]):
        # print('\n# Start to augment the Lable %d #' % i)
        data = stratified_data[i]
        X = data[:, 1:]
        if len(X) == max_samples:
            continue
        Y = data[:, 0]
        # get the time steps and the number of coefficients
        n_time_steps = len(X[0])

        # discrete fourier transform
        X_dft = get_dft_coefs(X, n_time_steps)
        # The splitting phase
        # split the X_dft to n groups for combination
        if X_dft.shape[1] > n_group:
            split_X_dft = np.array_split(X_dft, n_group, axis=1)
        else:  # if the shape[1] less than n_group, n_group=2
            split_X_dft = np.array_split(X_dft, 2, axis=1)

        # The random sampling phase ---
        # randomly samples the example in each split_data with tha corresponding label
        # the number of the sampled_data are half the original size of dataset
        n_samples = len(X)
        gap = max_samples-n_samples
        factor = gap / n_samples
        total_sampled = []
        for _ in range(int(factor)):
            N_selected = int(n_samples * 1)
            N_list = np.arange(n_samples)
            sampled_X_dft = []
            for j in range(n_group):
                split_data = split_X_dft[j]

                # start to sample
                selected_idx = np.random.choice(N_list, N_selected, replace=False)  #

                sampled_X_dft.append(split_data[selected_idx])
            # concatenate the sampled data from the n_group
            sampled_X_dft = np.concatenate(sampled_X_dft, axis=1)
            total_sampled.append(sampled_X_dft)

        # 因为下取整，会有部分数量不够，还需在取样一次
        if int(factor) * n_samples < gap:
            N_selected = gap - int(factor) * n_samples
            N_list = np.arange(n_samples)
            sampled_X_dft = []
            for j in range(n_group):
                split_data = split_X_dft[j]
                # start to sample
                selected_idx = np.random.choice(N_list, N_selected, replace=False)  #
                sampled_X_dft.append(split_data[selected_idx])
            # concatenate the sampled data from the n_group
            sampled_X_dft = np.concatenate(sampled_X_dft, axis=1)
            total_sampled.append(sampled_X_dft)

        total_sampled = np.concatenate(total_sampled)
        print(total_sampled.shape)
        # print('Shape of the sampled_X_dft', sampled_X_dft.shape)

        # inverse the frequency domain into time domain with the same time step
        X_combined = np.fft.irfft(total_sampled, n_time_steps)
        # add the corresponding label
        Y = Y[0, np.newaxis].repeat(X_combined.shape[0]).reshape(X_combined.shape[0], 1)
        data_combined = np.concatenate((Y, X_combined), axis=1)
        # print('Shape of th combined data', data_combined.shape)
        data_augmented.append(data_combined)
    print('\n# Succeed to augment the data! #')
    data_augmented = np.concatenate(data_augmented)
    print('Shape of the augmented data', data_augmented.shape)
    return data_augmented



def get_max_samples(stratified_data):
    max_number = -1
    for i in range(len(stratified_data)):
        nb_samples = len(stratified_data[i])
        print(nb_samples)
        if nb_samples > max_number:
            max_number = nb_samples
    return max_number




if __name__ == '__main__':
    data_path = 'UCRArchive_2018/ProximalPhalanxTW/ProximalPhalanxTW_TRAIN.tsv'
    data, n_class = load_ucr(data_path)
    s_data = stratify_by_label(data)
    data_aug = data_aug_by_dft(s_data, n_group=2)
