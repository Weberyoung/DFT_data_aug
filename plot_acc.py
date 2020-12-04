import matplotlib.pyplot as plt
import numpy as np

ori = [84.3, 69.7, 86, 95.5, 90.5, 31.3, 33.9, 72.4, 100, 71, 69, 72.7, 88.9, 92.8, 94.6, 95.8, 71.8, 48, 60.8, 96.1,
       72.3, 97.7, 100, 96]
after = [84.6, 77.7, 86.4, 98, 91.3, 95.1, 37.2, 85.5, 100, 75.8, 66.7, 72.2, 88.6, 93.4, 94.4, 96.3, 74.9, 48.8, 65.6,
         96.4, 79.3, 95.2, 100, 97.4]
plt.scatter(ori,after, marker='x',c='black')
a=[0,100]
b=[0,100]
plt.plot(a,b,color='r',linewidth=2)
plt.grid(linewidth=0.3)
plt.xlabel('Accuracy of FCN')
plt.ylabel('Accuracy after Data Augmentation')
plt.text(60,20,'Win: 17 / Draw:2 / Lose: 6',c='b',weight='bold')
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()