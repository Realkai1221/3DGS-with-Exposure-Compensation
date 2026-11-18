"""Applying an exposure map to colours (Eq. 1 of the paper) and to depth."""

import torch


def apply_exposure(color_map, exposure_map):
    """Exposure-free colour -> colour as the camera saw it:  C = C_adj * e.

    This is Eq. 1 solved for C.  Training compares this with the untouched
    photograph.  Dividing the photograph by the map instead (the form below)
    lets the optimiser lower the loss simply by raising the exposure, because
    that shrinks both sides of the comparison; that is why v2 needed the tight
    [0.8, 1.5] clamp.
    """
    return color_map * exposure_map


def adjust_color_with_exposure(color_map, exposure_map, epsilon=1e-4):
    """Photograph -> exposure-free colour:  C_adj = C / (e + epsilon)   (Eq. 1).

    Kept for visualising the exposure-free ground truth; not used in the loss.
    """
    return color_map / (exposure_map + epsilon)


def adjust_depth_with_exposure(depth_map, exposure_map, method='sigmoid'):
    if method == 'sigmoid':
        weight = torch.sigmoid(5.0 * (exposure_map - 0.5))  # [B, 1, H, W]
    else:
        weight = torch.clamp(exposure_map, min=0.5, max=1.0)

    return depth_map * weight.squeeze(1)
