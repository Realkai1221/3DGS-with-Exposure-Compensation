"""Initial exposure of each Gaussian from the grey level it projects to."""

import torch
import torch.nn.functional as F


def rgb_to_gray(image):
    """Luminance of an RGB image shaped [B, 3, H, W]."""
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


@torch.no_grad()
def sample_gray_from_images(xyz, cameras, near=0.01):
    """Sum and count of the grey levels every 3D point projects to.

    xyz:     [N, 3] Gaussian centres.
    cameras: training cameras (need .original_image and .full_proj_transform).
    Returns (gray_sum [N], n_views [N]).

    A point counts for a view when it is in front of the camera and inside the
    image.  Occlusion is not tested, in the same way 3DGS takes the starting
    colour of a point from all its SfM observations.
    """
    device = xyz.device
    n = xyz.shape[0]
    gray_sum = torch.zeros(n, device=device)
    n_views = torch.zeros(n, device=device)
    xyz_h = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)

    for cam in cameras:
        image = cam.original_image.to(device)[:3]
        gray = rgb_to_gray(image[None])                     # [1, 1, H, W]
        # 3DGS stores its matrices transposed, so points multiply from the left.
        p = xyz_h @ cam.full_proj_transform.to(device)      # [N, 4]
        w = p[:, 3]
        ndc = p[:, :2] / (w[:, None] + 1e-7)
        inside = (w > near) & (ndc.abs() <= 1.0).all(dim=1)
        if not inside.any():
            continue
        grid = ndc[inside].view(1, -1, 1, 2)
        # align_corners=False matches the rasteriser's NDC -> pixel convention.
        sampled = F.grid_sample(gray, grid, mode="bilinear", padding_mode="border",
                                align_corners=False)
        gray_sum[inside] += sampled.view(-1)
        n_views[inside] += 1
    return gray_sum, n_views


@torch.no_grad()
def initialize_exposure_from_projection(xyz, cameras, default=0.95, low=0.1, high=1.0):
    """Mean grey level seen by each Gaussian, used as its starting exposure.

    Returns (e_init [N], seen [N] bool).  Gaussians that project into no training
    image keep `default`.  Values are kept inside [low, high], the same range the
    range loss asks for.
    """
    gray_sum, n_views = sample_gray_from_images(xyz, cameras)
    seen = n_views > 0
    e_init = torch.full_like(gray_sum, default)
    e_init[seen] = (gray_sum[seen] / n_views[seen]).clamp(low, high)
    return e_init, seen
