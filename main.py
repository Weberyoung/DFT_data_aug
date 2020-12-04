from dft_aug import data_aug_by_dft
from utils import UcrDataset, UCR_dataloader, load_ucr, stratify_by_label
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'UCRArchive_2018/Car/Car_TRAIN.tsv'
    data, n_class = load_ucr(path)
    print(data.shape)
    stratified_data = stratify_by_label(data)
    data_aug = data_aug_by_dft(stratified_data, ratio=0.4, n_group=4)
    data_ori_and_aug = np.concatenate((data, data_aug))
    dataset = UcrDataset(data_ori_and_aug)
    dataloader = UCR_dataloader(dataset, batch_size=148)
    for i, (data, label) in enumerate(dataloader):
        print(data.size())
        print(label)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
