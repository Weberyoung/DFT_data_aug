import numpy as np
import pandas as pd

data = pd.read_csv('UCR85.csv', header=None)
data = data.values

name = data[1:, 0]
arry = np.ndarray(0)
for n in name:
    cmd = 'nohup python train_resNet.py --run_tag %s --cuda > train_log/%s_wo_aug.txt &' % (n, n)
    arry = np.append(arry, cmd)
arry = arry.reshape(arry.shape[0], 1)
np.savetxt('train_resNet_wo_aug.txt', arry, fmt='%s')
# n_class = data[:,7]
# testsize =data[:,5]
# seqlen = data[:,6]
# factor = testsize/n_class
# #print(testsize)
# factor =[int(i) for i in factor]
# #print(factor)
#
# arry = np.ndarray(0)
#
# for (name,classes,fa,se) in zip(name,n_class,factor,seqlen):
#
#     arry = np.append(arry,cmd)
# arry = arry.reshape(arry.shape[0],1)
# np.savetxt('train_ResNet_test_cmd.txt',arry,fmt='%s')
