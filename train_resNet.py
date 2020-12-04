from models import ResNet, ConvNet
import torch.nn as nn
import argparse
from utils import UcrDataset, UCR_dataloader, load_ucr, stratify_by_label
from dft_aug import data_aug_by_dft
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import os
import random
import numpy as np
from constants import TOTAL85

parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true', help='')
parser.add_argument('--aug', action='store_true', help='')
parser.add_argument('--gpu', type=str, default='0', help='the index of test sample ')
parser.add_argument('--channel_last', type=bool, default=True, help='the channel of data is last or not')
parser.add_argument('--runs', type=int, default=3, help='the runs to calculate accuracy')
parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--checkpoints_folder', default='model_checkpoints', help='folder to save checkpoints')
parser.add_argument('--run_tag', default='ECG200', help='tags for the current run')
parser.add_argument('--model', default='f', help='the model type(ResNet,FCN)')
parser.add_argument('--n_group', type=int,  help='the number of group')

opt = parser.parse_args()

print(opt)
# configure cuda
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    print("You have a cuda device, so you might want to run with --cuda as option")

device = torch.device("cuda:0" if opt.cuda else "cpu")

# all_reprot_metrics = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
#                                   columns=['acc_mean', 'acc_std', 'F1_mean', 'F1_std'])
# all_reprot_metrics = all_reprot_metrics.drop(index=[0])
if opt.aug:
    dir = 'report_metrics/%s_aug_%s' % (opt.model, str(opt.n_group))
else:
    dir = 'report_metrics/%s' % opt.model


def train(name, data_aug=opt.aug):
    record = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float),
                          columns=['precision', 'accuracy', 'recall', 'F1'])
    for _ in range(opt.runs):
        seed = random.randint(1, 10000)
        print("Random Seed: ", seed)
        torch.manual_seed(seed)

        # mkdirs for checkpoints output
        os.makedirs(opt.checkpoints_folder, exist_ok=True)
        os.makedirs('%s/%s' % (opt.checkpoints_folder, name), exist_ok=True)
        os.makedirs('report_metrics', exist_ok=True)
        if opt.aug:
            root_dir = 'report_metrics/%s_aug_%s/%s' % (opt.model, str(opt.n_group), name)
            os.makedirs(root_dir, exist_ok=True)

        else:
            root_dir = 'report_metrics/%s/%s' % (opt.model, name)
            os.makedirs(root_dir, exist_ok=True)

        # 加载数据集
        path = 'UCRArchive_2018/' + name + '/' + name + '_TRAIN.tsv'
        train_set, n_class = load_ucr(path)
        train_size = len(train_set)

        if data_aug:
            print('启用数据增强！')
            stratified_train_set = stratify_by_label(train_set)
            data_aug_set = data_aug_by_dft(stratified_train_set, opt.n_group, train_size)
            total_set = np.concatenate((train_set, data_aug_set))
            print('Shape of total set', total_set.shape)
            dataset = UcrDataset(total_set, channel_last=opt.channel_last)
        else:
            dataset = UcrDataset(train_set, channel_last=opt.channel_last)

        batch_size = int(min(len(dataset) / 10, 16))
        dataloader = UCR_dataloader(dataset, batch_size)

        # Common behavior
        seq_len = dataset.get_seq_len()  # 初始化序列长度
        # 创建分类器对象\损失函数\优化器
        if opt.model == 'r':
            net = ResNet(n_in=seq_len, n_classes=n_class).to(device)
        if opt.model == 'f':
            net = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(net.parameters(), lr=opt.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                         min_lr=0.0001)

        min_loss = 10000
        print('############# Start to Train ###############')
        net.train()
        for epoch in range(opt.epochs):
            for i, (data, label) in enumerate(dataloader):
                data = data.float()
                data = data.to(device)
                label = label.long()
                label = label.to(device)
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, label.view(label.size(0)))
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                # print('[%d/%d][%d/%d] Loss: %.8f ' % (epoch, opt.epochs, i + 1, len(dataloader), loss.item()))
            if loss < min_loss:
                min_loss = loss
                # End of the epoch,save model
                print('MinLoss: %.10f Saving the best epoch model.....' % min_loss)
                torch.save(net, '%s/%s/%s_%s_best.pth' % (opt.checkpoints_folder, name, opt.model, str(opt.n_group)))
        net_path = '%s/%s/%s_%s_best.pth' % (opt.checkpoints_folder, name, opt.model, str(opt.n_group))
        one_record = eval_accuracy(net_path, name)
        print('The minimum loss is %.8f' % min_loss)
        record = record.append(one_record, ignore_index=True)
    record = record.drop(index=[0])
    record.loc['mean'] = record.mean()
    record.loc['std'] = record.std()
    record.to_csv(root_dir + '/metrics.csv')
    # all_reprot_metrics.loc[name, 'acc_mean'] = record.at['mean', 'accuracy']
    # all_reprot_metrics.loc[name, 'acc_std'] = record.at['std', 'accuracy']
    # all_reprot_metrics.loc[name, 'F1_mean'] = record.at['mean', 'F1']
    # all_reprot_metrics.loc[name, 'F1_std'] = record.at['std', 'F1']

    print('\n')


def calculate_metrics(y_true, y_pred):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'F1'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['F1'] = f1_score(y_true, y_pred, average='macro')

    return res


def eval_accuracy(net_path, name):
    path = 'UCRArchive_2018/' + name + '/' + name + '_TEST.tsv'
    test_set, n_class = load_ucr(path)
    x_test = torch.tensor(test_set[:, 1:]).float().to(device)
    y_test = torch.tensor(test_set[:, 0]).long()
    net = torch.load(net_path)
    net.eval()
    y_pred = net(x_test).cpu()
    y_pred = torch.argmax(y_pred, dim=1)
    res = calculate_metrics(y_test, y_pred)
    return res


if __name__ == '__main__':
    if opt.run_tag == 'all':
        for n in TOTAL85:
            train(name=n)
    else:
        train(name=opt.run_tag)
    # all_reprot_metrics.to_csv(dir + '/all_metrics.csv')

