import cv2
import torch
from .custom_image_param import custom_image_param
from .transforms import imagenet_transform
from .utils import (
    fft_to_rgb_custom_img,
    lucid_colorspace_to_rgb, 
    init_image_param, 
    fft_to_rgb,
    chw_rgb_to_fft_param
)


class masked_image_param(custom_image_param):
    def __init__(self, mask_tensor, image = None , device = 'cuda'):
        """Custom image param, but with a mask over the original image. 

        The mask helps update only certain parts of the image 
        and leave the rest untouched by training. Ideal for 
        things like: "differentiable" backgrounds.

        Args:
            image (str or torch.tensor): "path/to/image.jpg" or NCHW tensor
            mask_tensor (torch.tensor): NCHW tensor whose values are clipped between 0,1
            device (str): 'cpu' or 'cuda'
        """

        self.width = mask_tensor.shape[-1]
        self.height = mask_tensor.shape[-2]

        if image is None:
            if self.width %2 ==1:
                self.param = init_image_param(height = self.height, width = self.width + 1, sd = 0.01, device = device)
            else:
                self.param = init_image_param(height = self.height, width = self.width, sd = 0.01, device = device)
            img = fft_to_rgb(height = self.height, width = self.width,  image_parameter = self.param, device= device)
            img = lucid_colorspace_to_rgb(t = img, device = device)
            image = torch.sigmoid(img)


        super().__init__(image = image, device= device)

        self.mask = mask_tensor.to(self.device)

        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)/255.
            image = torch.tensor(image).permute(-1,0,1).unsqueeze(0)

        self.original_nchw_image_tensor = image.to(device)

        assert self.mask.shape[-2:] == self.to_nchw_tensor(device = self.device).shape[-2:], "The height and width of the input image" + str(self.to_nchw_tensor(device = self.device).shape[-2:]) + "and the mask" + str(self.mask.shape[-2:]) + "do not match."

    def to_chw_tensor(self, device = 'cpu'):
        t = self.forward(device= device).squeeze(0).clamp(0,1).detach()  + self.original_nchw_image_tensor.to(device) * (1-self.mask.to(device)) 
        return t.squeeze(0)

    def get_spatial_grads(self, device):

        out = fft_to_rgb_custom_img(height = self.height, width = self.width, image_parameter= self.param.grad.data, device= device)
        out = self.normalize(lucid_colorspace_to_rgb(t = out, device= device).clamp(0,1), device = self.device)
        return out
    
    def apply_mask_on_grads(self):
        
        self.param.grad.data = chw_rgb_to_fft_param(
            x = (self.get_spatial_grads(self.device)  * self.mask).squeeze(0) , 
            device = self.device
        )


    def forward(self, device):
        return self.normalize(self.postprocess(device = device), device= device).clamp(0,1) * self.mask.to(device) 

    def update_mask(self, mask):
        """updates the mask to have a new value

        Warning: it also updates the original image 

        Args:
            mask (torch.tensor): new mask to be used
        """

        self.original_nchw_image_tensor = self.to_chw_tensor(device = self.device).unsqueeze(0)
        self.mask = mask.to(self.device)


