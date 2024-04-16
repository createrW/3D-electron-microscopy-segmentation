from torchviz import make_dot
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
from my_3D.unet3d_2 import *
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(1, 1, 1)
# model = model.to(device)
g = make_dot(model(torch.rand(1, 1, 128, 128, 128)),params=dict(model.named_parameters()))
g.view()