from models import ResNet, ConvNet
import torch.nn as nn
import argparse
from utils import UcrDataset, UCR_dataloader, load_ucr, stratify_by_label
from dft_aug import data_aug_by_dft
import torch.optim as optim
import torch.utils.data
import os
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true', help='')
parser.add_argument('--aug', action='store_true', help='')
parser.add_argument('--query_one', action='store_true', help='query the probability of  target idx sample')
parser.add_argument('--idx', type=int, help='the index of test sample ')
parser.add_argument('--gpu', type=str, default='0', help='the index of test sample ')
parser.add_argument('--channel_last', type=bool, default=True, help='the channel of data is last or not')
# parser.add_argument('--n_class', type=int, default=2, help='the class number of dataset')
parser.add_argument('--epochs', type=int, default=1500, help='number of epochs to train for')
parser.add_argument('--e', default=1499, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--checkpoints_folder', default='model_checkpoints', help='folder to save checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--run_tag', default='ECG200', help='tags for the current run')
parser.add_argument('--model', default='f', help='the model type(ResNet,FCN)')
parser.add_argument('--checkpoint_every', default=5, help='number of epochs after which saving checkpoints')
parser.add_argument('--ratio', type=float, default=0.4, help='the split ratio')
parser.add_argument('--n_group', type=int, default=4, help='the number of group')

opt = parser.parse_args()

print(opt)
# configure cuda
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    print("You have a cuda device, so you might want to run with --cuda as option")

device = torch.device("cuda:0" if opt.cuda else "cpu")
# Create randow seed for cpu/GPU ##############
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


def train(data_aug=opt.aug):
    # mkdirs for checkpoints output
    os.makedirs(opt.checkpoints_folder, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_folder, opt.run_tag), exist_ok=True)

    # 加载数据集 #
    path = '85_UCRArchive/' + opt.run_tag + '/' + opt.run_tag + '_TRAIN.tsv'
    train_set, n_class = load_ucr(path)

    if data_aug:
        print('启用数据增强！')
        stratified_train_set = stratify_by_label(train_set)
        data_aug_set = data_aug_by_dft(stratified_train_set, opt.ratio, opt.n_group)
        total_set = np.concatenate((train_set, data_aug_set))
        print('Shape of total set', total_set.shape)
        dataset = UcrDataset(total_set, channel_last=opt.channel_last)
    else:
        dataset = UcrDataset(train_set, channel_last=opt.channel_last)

    batch_size = int(min(len(dataset) / 10, 16))
    print('dataset length: ', len(dataset))
    print('batch size：', batch_size)
    dataloader = UCR_dataloader(dataset, batch_size)

    # Common behavior
    seq_len = dataset.get_seq_len()  # 初始化序列长度

    print('序列长度:', seq_len)

    # 创建分类器对象\损失函数\优化器
    if opt.model == 'r':
        net = ResNet(n_in=seq_len, n_classes=n_class).to(device)
    if opt.model == 'f':
        net = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
    net.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    print('############# Start to Train ###############')
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
            print('[%d/%d][%d/%d] Loss: %.4f ' % (epoch, opt.epochs, i + 1, len(dataloader), loss.item()))
        eval_accuracy(net=net)
        # End of the epoch,save model
        if (epoch % (opt.checkpoint_every * 10) == 0) or (epoch == (opt.epochs - 1)):
            print('Saving the %dth epoch model.....' % epoch)
            torch.save(net, '%s/%s/%s%depoch.pth' % (opt.checkpoints_folder, opt.run_tag, opt.model, epoch))


def eval_accuracy(net):
    path = '85_UCRArchive/' + opt.run_tag + '/' + opt.run_tag + '_TEST.tsv'
    test_set, n_class = load_ucr(path)
    dataset = UcrDataset(test_set, channel_last=opt.channel_last)
    batch_size = int(min(len(dataset) / 10, 16))
    dataloader = UCR_dataloader(dataset, batch_size)
    with torch.no_grad():
        net.eval()
        total = 0
        correct = 0
        for i, (data, label) in enumerate(dataloader):
            # 处理数据的 device（cpu/gpu）？
            data = data.float()
            data = data.to(device)
            label = label.long()
            label = label.to(device)
            label = label.view(label.size(0))
            # 总的数据个数
            total += label.size(0)
            # 预测值
            out = net(data)
            softmax = nn.Softmax(dim=-1)  # 归一化概率
            prob = softmax(out)  # 概率值向量
            pred_label = torch.argmax(prob, dim=1)  # 取概率最大的维度
            correct += (pred_label == label).sum().item()

        print('The EVAL Accuracy of %s is :  %.2f %%' % (path, correct / total * 100))
    return correct / total * 100


def test():
    data_path = '85_UCRArchive/' + opt.run_tag + '/' + opt.run_tag + '_TEST.tsv'
    dataset = UcrDataset(data_path, channel_last=opt.channel_last)
    batch_size = int(min(len(dataset) / 10, 16))
    print('dataset length: ', len(dataset))
    print('batch_size:', batch_size)
    dataloader = UCR_dataloader(dataset, batch_size)
    # 加载模型
    type = opt.model
    model_path = 'model_checkpoints/' + opt.run_tag + '/' + type + str(opt.e) + 'epoch.pth'
    model = torch.load(model_path, map_location='cpu')

    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        for i, (data, label) in enumerate(dataloader):
            # 处理数据的 device（cpu/gpu）？
            data = data.float()
            data = data.to(device)
            label = label.long()
            label = label.to(device)
            label = label.view(label.size(0))
            # 总的数据个数
            total += label.size(0)
            # 预测值
            out = model(data)
            softmax = nn.Softmax(dim=-1)  # 归一化概率
            prob = softmax(out)  # 概率值向量
            pred_label = torch.argmax(prob, dim=1)  # 取概率最大的维度

            correct += (pred_label == label).sum().item()

        print('The TEST Accuracy of %s is :  %.2f %%' % (data_path, correct / total * 100))


def query_one(idx):
    # 加载数据集
    data_path = '85_UCRArchive/' + opt.run_tag + '/' + opt.run_tag + '_TEST.txt'
    test_data = np.loadtxt(data_path)
    test_data = torch.from_numpy(test_data)

    test_one = test_data[idx]  # 获得某一行索引为idx的时间序列

    # 获得数据和标签
    X = test_one[1:].float()
    X = X.to(device)
    y = test_one[0].long() - 1
    y = y.to(device)
    if y < 0:
        y = opt.n_class - 1
    print('真实标签为：', y)
    # 加载训练好的模型
    type = opt.model
    model_path = 'model_checkpoints/' + opt.run_tag + '/' + type + str(opt.e) + 'epoch.pth'
    model = torch.load(model_path, map_location='cpu')
    model.eval()

    # 获得概率输出
    out = model(X)
    softmax = nn.Softmax(dim=-1)  # 归一化概率
    prob_vector = softmax(out)
    print('概率向量：', prob_vector)
    prob = prob_vector.view(opt.n_class)[y].item()

    # print(prob)
    print('Confidence in true class of the %d sample is  %.4f ' % (idx, prob))


if __name__ == '__main__':
    if opt.test:
        test()
    elif opt.query_one:
        query_one(opt.idx)
    else:
        train()
