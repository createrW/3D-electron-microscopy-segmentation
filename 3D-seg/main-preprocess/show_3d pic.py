from dataset3d import *
import matplotlib.pyplot as plt
import numpy as np
dataset = EMdataset3d(mode = 'test')
i = np.random.randint(0,len(dataset))
data = dataset[i]
#     print(data["volume"].shape)
#     print(data["volume"])
#     plt.subplot(121)
#     plt.imshow(np.mean(data["volume"].squeeze(), axis=0))
#     plt.subplot(122)
#     plt.imshow(np.mean(data["target"].squeeze(), axis=0))
#     plt.show()
image = data["volume"]
mask = data["target"]
plt.subplot(133)
