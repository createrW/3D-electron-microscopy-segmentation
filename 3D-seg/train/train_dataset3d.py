from preprocess.dataset3d import *
import matplotlib.pyplot as plt
train_dataset = EMdataset3d(mode = 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
print(len(train_loader))
i = np.random.randint(0,len(train_dataset))
data = train_dataset[i]
image = data["volume"][0]
mask = data["target"][0]
print(image.shape)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image[64, :, :], cmap='bone')
axes[0].set_title('YZ Plane')

axes[1].imshow(image[:, 64, :], cmap='bone')
axes[1].set_title('XZ Plane')

axes[2].imshow(image[:, :, 64], cmap='bone')
axes[2].set_title('XY Plane')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(mask[64, :, :], cmap='gray')
axes[0].set_title('YZ Plane')

axes[1].imshow(mask[:, 64, :], cmap='gray')
axes[1].set_title('XZ Plane')

axes[2].imshow(mask[:, :, 64], cmap='gray')
axes[2].set_title('XY Plane')
plt.show()
