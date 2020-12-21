import numpy as np
import matplotlib.pyplot as plt
from utils import load_ucr, stratify_by_label
from sklearn import manifold
from  dft_aug import data_aug_by_dft

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
    colors = ['orange', 'b', 'r', 'y', 'black', 'orange', 'm']
    markers = ['s', 'o','^','d','x','+']
    X_tsne = tsne.fit_transform(X)
    X_tsne = normalization(X_tsne)
    for i in range(X_tsne.shape[0]):
        if i < split:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=colors[int(Y[i])], s=80, marker=markers[int(Y[i])], alpha=0.6)
            # if Y[i] == 1:
            #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='red', s=80, marker='o', alpha=1)
            # else:
            #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='blue', s=80, marker='o', alpha=1)
        else:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='', edgecolors=colors[int(Y[i])], s=80, marker=markers[int(Y[i])], alpha=0.6)
            # if Y[i] == 1:
            #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='orangered', s=80, marker='x', alpha=1)
            # else:
            #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c='lightblue', s=80, marker='x', alpha=1)
    plt.xlabel('x_tsne')
    plt.ylabel('y_tsne')
    # plt.savefig('tsne.pdf')
    plt.show()


def plot_ori_tsne(X, Y):
    tsne = manifold.TSNE(n_components=2, perplexity=40, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    X_tsne = normalization(X_tsne)
    colors = ['orange', 'b', 'r', 'y', 'black', 'orange', 'm']

    markers = ['s', 'o','^','d','x','+']
    for i in range(X_tsne.shape[0]):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=colors[int(Y[i])], s=80, marker=markers[int(Y[i])], alpha=0.7)
    plt.xlabel('x_tsne')
    plt.ylabel('y_tsne')

    # plt.savefig('tsne.pdf')
    plt.show()


if __name__ == '__main__':
    data_path = 'UCRArchive_2018/Car/Car_TRAIN.tsv'
    data, n_class = load_ucr(data_path)
    x_ori = data[:, 1:]
    y_ori = data[:, 0]
    time_step = len(data[0]) - 1
    stratified_data = stratify_by_label(data)
    data_aug = data_aug_by_dft(stratified_data, n_group=4, train_size=len(data))
    x_aug = data_aug[:, 1:]
    y_aug = data_aug[:, 0]
    plot_tsne(x_ori, x_aug, y_ori, y_aug,n_class=n_class)
    plot_ori_tsne(x_ori, y_ori)