import torch
from .constants import circuits_default_config

class index_register():
    def __init__(self, dreamer, config, top_k = 5):

        self.indices = []
        self.values = []
        self.top_k = top_k
        self.config = circuits_default_config
        self.config["custom_func"] = self.custom_func
        self.config["layers"] = config["layers"]
        self.dreamer = dreamer

        if "image" in list(config.keys()):
            self.config["image"] = config["image"]

        if "image_path"  in list(config.keys()):
            self.config["image_path"] = config["image_path"]

    def store(self, layer_outputs):
        for layer_output_tensor in layer_outputs:
            indices, values = self.find_top_k_indices(layer_output_tensor)
            self.indices.append(indices)
            self.values.append(values)

    def custom_func(self, layer_outputs):
        self.store(layer_outputs)
            
    def find_top_k_indices(self, layer_outputs_tensor):

        means = torch.mean(layer_outputs_tensor, axis = (1,2))  ##  mean per channel 
        values, indices = torch.topk(means, k = self.top_k, dim = 0)
        return indices, values
    
    def find_activations(self):
        try:
            out = self.dreamer.deep_dream(self.config)
        except AttributeError:
            pass