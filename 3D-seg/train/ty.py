from monai.networks.nets import UNet
import torch
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 128
x = torch.Tensor(1, 1, image_size, image_size, image_size)
x.to(device)
print("x size: {}".format(x.size()))

out = model(x)
print("out size: {}".format(out.size()))
