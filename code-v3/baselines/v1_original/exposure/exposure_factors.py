import torch
import torch.nn as nn
import torch.nn.functional as F


class ExposureModel(torch.nn.Module):

    def __init__(self, num_gaussians, init_value=0.8):
        super().__init__()
        self.epsilon = 1e-4
        self.exposure = torch.nn.Parameter(
            torch.ones(num_gaussians) * init_value
        )
        
        
    def forward(self, ids):
        
        return torch.clamp(self.exposure[ids], min=0.05, max=1.5)

    def get_all(self):
        return torch.clamp(self.exposure, min=0.05, max=1.5)

def rgb_to_gray(image):
   
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def sample_gray_from_images(gray_images, projection_coords):
    
    voxel_values = [[] for _ in range(projection_coords[0].shape[0])]

    for img, coords in zip(gray_images, projection_coords):
        grid = coords.view(1, -1, 1, 2)  
        sampled = F.grid_sample(img, grid, align_corners=True)
        sampled = sampled.view(-1)  
        for i in range(sampled.shape[0]):
            voxel_values[i].append(sampled[i].item())

    return voxel_values  

def initialize_exposure_from_projection(rgb_images, projection_coords):

    gray_images = [rgb_to_gray(img) for img in rgb_images]  
    voxel_gray_values = sample_gray_from_images(gray_images, projection_coords)

    N = len(voxel_gray_values)
    e_init = torch.zeros(N)
    for i in range(N):
        if len(voxel_gray_values[i]) > 0:
            e_init[i] = sum(voxel_gray_values[i]) / len(voxel_gray_values[i])
        else:
            e_init[i] = 0.8  

    return e_init
