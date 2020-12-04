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


def data_aug_by_dft(stratified_data, n_group):
    """
    :param stratified_data: the data have been stratified by the corresponding label
    :param n_group:
    :return:
    """
    np.random.seed(66)
    data_augmented = []
    # deal with each label
    for i in range(stratified_data.shape[0]):
        # print('\n# Start to augment the Lable %d #' % i)
        data = stratified_data[i]
        print(len(data))
        X = data[:, 1:]
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
        sampled_X_dft = []
        for j in range(n_group):
            split_data = split_X_dft[j]
            n_samples = split_data.shape[0]
            N_selected = int(n_samples * 1)
            N_list = np.arange(n_samples)
            # start to sample

            selected_idx = np.random.choice(N_list, N_selected, replace=False)  #
            #print(selected_idx)
            sampled_X_dft.append(split_data[selected_idx])

        # concatenate the sampled data from the n_group
        sampled_X_dft = np.concatenate(sampled_X_dft, axis=1)
        # print('Shape of the sampled_X_dft', sampled_X_dft.shape)

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


def normalization(X_tsne):
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    return X_norm


def plot_tsne(x, x_aug, y, y_aug, n_class):
    np.random.seed(2)
    tsne = manifold.TSNE(n_components=2, perplexity=40, n_iter=500)
    split = x.shape[0]
    X = np.concatenate((x, x_aug), axis=0)
    Y = np.concatenate((y, y_aug), axis=0)
    print(X.shape, Y.shape)
    colors = ['b', 'r', 'g', 'y', 'black', 'orange']
    X_tsne = tsne.fit_transform(X)
    X_tsne = normalization(X_tsne)
    for i in range(X_tsne.shape[0]):
        if i < split:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=colors[int(Y[i])], s=80, marker='o', alpha=1)
            # if Y[i] == 1:
            #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='red', s=80, marker='o', alpha=1)
            # else:
            #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='blue', s=80, marker='o', alpha=1)
        else:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=colors[int(Y[i])], s=80, marker='^', alpha=1)
            # if Y[i] == 1:
            #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='orangered', s=80, marker='x', alpha=1)
            # else:
            #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='lightblue', s=80, marker='x', alpha=1)
    plt.xlabel('x_tsne')
    plt.ylabel('y_tsne')
    # plt.legend(['1',
    #             '2',
    #             '1+', '2+'], loc='best')
    # plt.savefig('tsne.pdf')
    plt.show()


def plot_ori_tsne(X, Y):
    tsne = manifold.TSNE(n_components=2, perplexity=40, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    X_tsne = normalization(X_tsne)
    colors = ['b', 'r', 'g', 'y', 'black', 'orange']

    for i in range(X_tsne.shape[0]):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=colors[int(Y[i])], s=80, marker='o', alpha=1)
    plt.xlabel('x_tsne')
    plt.ylabel('y_tsne')

    # plt.savefig('tsne.pdf')
    plt.show()


if __name__ == '__main__':
    data_path = 'UCRArchive_2018/Earthquakes/Earthquakes_TRAIN.tsv'
    data, n_class = load_ucr(data_path)
    x_ori = data[:, 1:]
    y_ori = data[:, 0]
    time_step = len(data[0]) - 1
    stratified_data = stratify_by_label(data)
    data_aug = data_aug_by_dft(stratified_data, 5)
    # x_aug = data_aug[:, 1:]
    # y_aug = data_aug[:, 0]
    # plot_tsne(x_ori, x_aug, y_ori, y_aug,n_class=n_class)
    # plot_ori_tsne(x_ori, y_ori)
    # x = data_aug[0, 1:]
    # plt.plot(stratified_data[0][6, 1:], color='b')
    # plt.plot(x, color='r')
    # plt.legend(['Original', 'New'])
    # plt.title('N_Group = 5')
    # plt.show()
    # print(n_class)
