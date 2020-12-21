import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.approximation import DiscreteFourierTransform
from utils import load_ucr, stratify_by_label
import random
from sklearn import manifold


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


def data_aug_by_dft(stratified_data, n_group, train_size):
    """
    :param stratified_data: the data have been stratified by the corresponding label
    :param n_group:
    :return:
    """
    # np.random.seed(66)
    data_augmented = []
    # deal with each label
    for i in range(stratified_data.shape[0]):
        # print('\n# Start to augment the Lable %d #' % i)
        data = stratified_data[i]
        X = data[:, 1:]
        Y = data[:, 0]
        # get the time steps and the number of coefficients
        n_time_steps = len(X[0])

        # discrete fourier transform
        X_dft = get_dft_coefs(X, n_time_steps)
        # The splitting phase
        # split the X_dft to n groups for combination

        n_group_real = X_dft.shape[1]
        split_X_dft = np.array_split(X_dft, n_group_real, axis=1)

        # The random sampling phase ---
        # randomly samples the example in each split_data with tha corresponding label
        # the number of the sampled_data are half the original size of dataset
        sampled_X_dft = []
        for j in range(n_group_real):
            split_data = split_X_dft[j]
            n_samples = split_data.shape[0]
            if train_size <= 200:
                factor = 1.0
            elif 200 < train_size <= 1000:
                factor = 0.7
            elif train_size > 1000:
                factor = 0.3
            N_selected = int(n_samples * factor)
            N_list = np.arange(n_samples)
            # start to sample
            selected_idx = np.random.choice(N_list, N_selected, replace=False)  #
            sampled_X_dft.append(split_data[selected_idx])

        # concatenate the sampled data from the n_group
        sampled_X_dft = np.concatenate(sampled_X_dft, axis=1)
        # inverse the frequency domain into time domain with the same time step
        X_combined = np.fft.irfft(sampled_X_dft, n_time_steps)
        # add the corresponding label
        Y = Y[:sampled_X_dft.shape[0], np.newaxis]
        data_combined = np.concatenate((Y, X_combined), axis=1)
        # print('Shape of th combined data', data_combined.shape)
        data_augmented.append(data_combined)
    print('\n# Succeed to augment the data! #')
    data_augmented = np.concatenate(data_augmented)
    print('Shape of the augmented data', data_augmented.shape)
    return data_augmented


if __name__ == '__main__':
    data_path = 'UCRArchive_2018/ECG200/ECG200_TRAIN.tsv'
    data, n_class = load_ucr(data_path)
    x_ori = data[:, 1:]
    y_ori = data[:, 0]
    time_step = len(data[0]) - 1
    stratified_data = stratify_by_label(data)
    data_aug = data_aug_by_dft(stratified_data, n_group=-1, train_size=len(data))
    print(data_aug.shape)
    x = data_aug[0, 1:]
    plt.axis('off')
    plt.plot(stratified_data[0][8, 1:], color='b')
    plt.plot(x, color='r')
    plt.legend(['Original', 'New'])
    plt.title('N_Group = 5')
    plt.show()
