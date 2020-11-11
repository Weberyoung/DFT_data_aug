import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.approximation import DiscreteFourierTransform


def statify_by_label(X, )

def get_dft_coefs(X, n_coefs):
    """
    :param X: the training set  numpy array shape:[n_samples, n_time_steps]
    :param n_coefs: number of the coefficients
    :return: numpy array shape: [n_samples, n_coefs]
    """
    dft = DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=False,
                                   norm_std=False)
    X_dft = dft.fit_transform(X)
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

def data_aug_by_dft()


if __name__ == '__main__':
    data_path = '85_UCRArchive/ECG200/ECG200_'
    trainset_path = data_path + 'TRAIN.tsv'
    trainset = pd.read_csv(trainset_path, sep='\t')
    X = np.array(trainset)[:, 1:]
    Y = np.array(trainset)[:, 0]
    print(Y)
    n_samples = X.shape[0]
    n_timestamps = X.shape[1]
    n_coefs = 10

    X_dft_new = get_dft_coefs(X, n_coefs)
    print(X_dft_new.shape)
    X_dft_1 = X_dft_new[1]

    X_dft_2 = X_dft_new[2]
    X_dft_12 = np.concatenate((X_dft_1[:8], X_dft_2[7:]))
    print(X_dft_12)
    X_irfft1 = np.fft.irfft(X_dft_12, n_timestamps)
    plot_dft(X[1], X_irfft1)
    #
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


