from The_final_unet3d import *
import matplotlib.pyplot as plt
import numpy as np
dataset = train_dataset
i = np.random.randint(0,len(dataset))
data = dataset[i]
#     print(data["volume"].shape)
#     print(data["volume"])
#     plt.subplot(121)
#     plt.imshow(np.mean(data["volume"].squeeze(), axis=0))
#     plt.subplot(122)
#     plt.imshow(np.mean(data["target"].squeeze(), axis=0))
#     plt.show()
image = data["image"].cpu().numpy()
mask = data["mask"].cpu().numpy()
image = np.transpose(image, (2, 0, 1))
mask = np.transpose(mask, (2, 0, 1))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# print(image[64, :, :].shape)
axes[0].imshow(image[64, :, :], cmap='bone')
axes[0].set_title('YZ Plane')

axes[1].imshow(image[:, 64, :], cmap='bone')
axes[1].set_title('XZ Plane')

axes[2].imshow(image[:, :, 64], cmap='bone')
axes[2].set_title('XY Plane')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# print(image[64, :, :].shape)
axes[0].imshow(mask[64, :, :], cmap='bone')
axes[0].set_title('YZ Plane')

axes[1].imshow(mask[:, 64, :], cmap='bone')
axes[1].set_title('XZ Plane')

axes[2].imshow(mask[:, :, 64], cmap='bone')
axes[2].set_title('XY Plane')

plt.tight_layout()
plt.show()