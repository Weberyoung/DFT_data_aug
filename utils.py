import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_ucr(path):
    data = pd.read_csv(path, sep='\t', header=None)
    data = np.array(data)  # read the data
    le = preprocessing.LabelEncoder()  # employ the labelEncoder
    ori_lable = data[:, 0]
    normal_lable = le.fit_transform(ori_lable)
    data[:, 0] = normal_lable
    return data  # all labels are transformed to [0 - n_classes-1]


def stratify_by_label(dataset):
    # Make all the same label data cluster
    labels = dataset[:, 0]
    order_indices = np.argsort(labels)
    ordered_data = dataset[order_indices]
    # ordered_labels = labels[order_indices]

    # Get data of each label
    n_classes = len(np.unique(labels))
    spilt_points = 0
    stratified_data = []
    for i in range(n_classes):
        label_sum = (labels == i).sum()
        stratified_data.append(ordered_data[spilt_points:spilt_points + label_sum])
        spilt_points += (labels == i).sum()
    stratified_data = np.array(stratified_data)

    return stratified_data, n_classes


if __name__ == '__main__':
    data = load_ucr('85_UCRArchive/ECG200/ECG200_TRAIN.tsv')
    stratified_data, n_class = stratify_by_label(data)
    assert (stratified_data.shape[0] == n_class)
    for i in range(n_class):
        print(stratified_data[i])
