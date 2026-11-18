# Stand-alone exposure model and colour adjustment used by trainv1.py

import torch


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


def adjust_color_with_exposure(color_map, exposure_map, epsilon=1e-4):

    adjustment_factor = 1.0 / (exposure_map + epsilon)
    adjustment_factor = torch.clamp(adjustment_factor, min=0.8, max=1.5)

    return color_map * adjustment_factor
