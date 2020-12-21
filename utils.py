import pandas as pd
from sklearn import preprocessing
import torch
import numpy as np
from torch.utils.data import Dataset


def load_ucr(path):
    data = pd.read_csv(path, sep='\t', header=None)
    data = data.values
    # data = np.array(data)  # read the data
    le = preprocessing.LabelEncoder()  # employ the labelEncoder
    ori_label = data[:, 0]
    n_class = len(np.unique(ori_label))
    normal_label = le.fit_transform(ori_label)
    data[:, 0] = normal_label
    return data, n_class  # all labels are transformed to [0 - n_classes-1]


class UcrDataset(Dataset):
    def __init__(self, data,channel_last=True):
        '''
         实现初始化方法，在初始化的时候将数据读载入
        :param data_path: 文件路径
        :param channel_last: 数据维度是否在最后一维
        '''
        self.data = data
        self.channel_last = channel_last
        if self.channel_last:
            self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1], 1])

        else:
            self.data = np.reshape(self.data, [self.data.shape[0], 1, self.data.shape[1]])

    def __len__(self):
        '''
        返回data的长度
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        根据 idx 返回一行数据
        UCR数据库的数据集类标在第一列，分别返回数据和类标
        '''
        if not self.channel_last:
            return self.data[idx, :, 1:], self.data[idx, :, 0]
        else:

            return self.data[idx, 1:, :], self.data[idx, 0, :]


    def get_seq_len(self):
        if self.channel_last:
            return self.data.shape[1] - 1
        else:
            return self.data.shape[2] - 1


def UCR_dataloader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0, drop_last=False)

    return data_loader


def stratify_by_label(dataset):
    """
    :param dataset: the data
    :return: the stratified data by the label
    """
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

    return stratified_data


if __name__ == '__main__':
    data, n_class = load_ucr('UCRArchive_2018/ECG200/ECG200_TRAIN.tsv')
    print(len(data))
    stratified_data = stratify_by_label(data)
    # stratified_data = np.concatenate(stratified_data)
    print(type(stratified_data))
    # assert (stratified_data.shape[0] == n_class)
    print(stratified_data.shape)
