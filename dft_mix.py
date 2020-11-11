import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.approximation import DiscreteFourierTransform
from utils import load_ucr, stratify_by_label

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


def plot_dft(X, X_irfft):
    plt.figure(figsize=(6, 4))
    plt.plot(X, 'o--', ms=4, label='Original')
    plt.plot(X_irfft, 'o--', ms=4, label='DFT - {0} coefs'.format(n_coefs))
    plt.legend(loc='best', fontsize=10)
    plt.xlabel('Time', fontsize=14)
    plt.title('Discrete Fourier Transform', fontsize=16)
    plt.show()


def data_aug_by_dft(stratified_data, spilt_ratio, n_group):
    for i in range(stratified_data.shape[0]):
        data = stratified_data[i]
        X = data[:, 1:]
        n_time_setps = len(X[0])
        n_coefs = int(n_time_setps * spilt_ratio)
        X_dft = get_dft_coefs(X, n_coefs)
        print(X_dft.shape)


if __name__ == '__main__':
    data_path = '85_UCRArchive/ECG200/ECG200_TRAIN.tsv'
    data = load_ucr(data_path)
    stratified_data, n_class = stratify_by_label(data)
    data_aug_by_dft(stratified_data, 0.20)

    # n_samples, n_timestamps = 1, 48
    #
    # # Toy dataset
    # rng = np.random.RandomState(42)
    # X = rng.randn(n_samples, n_timestamps)
    # # DFT transformation
    # n_coefs = 48
    # dft = DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=False,
    #                                norm_std=False)
    # X_dft = dft.fit_transform(X)
    # print(X_dft)
    # # Compute the inverse transformation
    # X_dft_new = get_dft_coefs(X_dft, n_coefs)
    #
    # print(X_dft_new)
    #
    # X_irfft = np.fft.irfft(X_dft_new[:,5:10], n_timestamps)
    # # Show the results for the first time series
    # plot_dft(X, X_irfft)
