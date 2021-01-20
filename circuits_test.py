import cv2
import numpy as np
import torch 
import matplotlib.pyplot as plt

from torch_dreams.dreamer import dreamer
import torchvision.models as models
from torch_dreams.circuits import index_register

"""
find indices of channels with max activations
"""
model = models.googlenet(pretrained=True)
dreamy_boi = dreamer(model)

layers_to_use = [
    model.inception3b,
    model.inception4a,
    model.inception4b,
    model.inception4c,
    model.inception4d,
    model.inception4e
]

"""
determining which channels get activated the most
"""

config = {
    "image_path": "images/zebra.jpeg",  ## german_shepherd.jpg
    "layers": layers_to_use,
    "octave_scale": 1.1,
    "num_octaves": 10,
    "iterations": 100,
    "lr": 0.07,
    "custom_func": None,
    "max_rotation": 0.9,
    "grayscale": False,
    "gradient_smoothing_coeff": None,
    "gradient_smoothing_kernel_size": None
}

reg = index_register(dreamy_boi, config, top_k= 3)
reg.find_activations()  ## saved into reg.indices

# for i in reg.values:
#     plt.plot(i.cpu().detach())
#     plt.show()

"""
run dreamer.deep_dream() on each channel that got activated, but on noise
"""
def select_channels(layer_idx = 0, channel_idx = []): 
    def custom_func(layer_outputs):
        # print(channel_idx)
        loss = layer_outputs[layer_idx][channel_idx].mean()
        return loss
    return custom_func

config["image_path"] = "images/noise.jpg"
all_vis = []
print(reg.indices)
print(reg.values)
for i in range(len(reg.indices)):
    config["custom_func"] = select_channels(layer_idx= i, channel_idx = reg.indices[i])
    out = dreamy_boi.deep_dream(config)
    all_vis.append(out)

"""
visualizing the composition 
"""
fig, ax = plt.subplots(nrows= 1, ncols= len(all_vis))
for i in range(len(reg.indices)):
    ax.flat[i].imshow(all_vis[i])

plt.show()