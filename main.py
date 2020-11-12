from dft_aug import data_aug_by_dft
from utils import UcrDataset, UCR_dataloader, load_ucr, stratify_by_label


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = '85_UCRArchive/ECG200/ECG200_TRAIN.tsv'
    data = load_ucr(path)
    stratified_data,_ = stratify_by_label(data)
    data_aug = data_aug_by_dft(stratified_data, ratio=0.4, n_group=4)
    dataset = UcrDataset(data_aug)
    dataloader = UCR_dataloader(dataset, batch_size=48)
    for i ,(data, label) in enumerate(dataloader):
        print(data.size())
        print(label)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
