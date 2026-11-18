import torch


def adjust_color_with_exposure(color_map, exposure_map, epsilon=1e-4):

    adjustment_factor = 1.0 / (exposure_map + epsilon)
    adjustment_factor = torch.clamp(adjustment_factor, min=0.8, max=1.5)

    return color_map * adjustment_factor


def adjust_depth_with_exposure(depth_map, exposure_map, method='sigmoid'):
    if method == 'sigmoid':
        weight = torch.sigmoid(5.0 * (exposure_map - 0.5))  # [B, 1, H, W]
    else:
        weight = torch.clamp(exposure_map, min=0.5, max=1.0)

    return depth_map * weight.squeeze(1)
