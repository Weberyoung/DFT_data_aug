import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#
# path = 'UCRArchive_2018/Beef/Beef_TRAIN.tsv'
# data = pd.read_csv(path, sep='\t', header=None)
# x = data.values[4, 1:]
# ax = plt.subplot()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.axis('off')
# ax.plot(x)
# plt.savefig('ecg.pdf', pad_inches=0.0, bbox_inches='tight')
# plt.show()



# x=np.arange(0,12*np.pi,0.01)
# y=np.cos(x)
#
# plt.axis('off')
# plt.plot(x,y)
# plt.savefig('12sin.pdf', pad_inches=0.0, bbox_inches='tight')
# plt.show()

sampleNo = 10000
mu = 0
sigma = 1
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo )
plt.hist(s, density=True)
plt.show()