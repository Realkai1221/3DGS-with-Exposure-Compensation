"""One call that combines the four exposure losses with configurable weights."""

import torch

from .exposure_losses import (
    exposure_rgb_loss,
    exposure_smooth_loss,
    exposure_range_loss,
    exposure_consistency_loss,
)

DEFAULT_LAMBDA_WEIGHTS = {
    'rgb': 1.0,
    'smooth': 0.1,
    'range': 0.01,
    'consistency': 0.05,
}


def init_exposure_supervision(
    image_with_exposure, ground_truth_image,
    gaussians,
    visible_idx,
    lambda_weights=None,
    smooth_samples=50000,
):
    """Total exposure loss for one training view.

    image_with_exposure: render x exposure map, [3, H, W].
    gaussians:           ExposureGaussianModel.
    visible_idx:         [M] indices of the Gaussians visible in this view.
    Terms whose weight is 0 are skipped.
    """
    if lambda_weights is None:
        lambda_weights = DEFAULT_LAMBDA_WEIGHTS

    exposure = gaussians.get_exposure
    zero = exposure.sum() * 0.0

    L_rgb = exposure_rgb_loss(image_with_exposure, ground_truth_image)

    L_smooth = zero
    if lambda_weights['smooth'] > 0 and gaussians.knn_idx is not None:
        sample_idx = visible_idx
        if sample_idx.numel() > smooth_samples:
            pick = torch.randint(0, sample_idx.numel(), (smooth_samples,), device=sample_idx.device)
            sample_idx = sample_idx[pick]
        L_smooth = exposure_smooth_loss(exposure, gaussians.knn_idx, sample_idx)

    L_range = zero
    if lambda_weights['range'] > 0 and visible_idx.numel() > 0:
        L_range = exposure_range_loss(gaussians._exposure[visible_idx])

    L_consistency = zero
    if lambda_weights['consistency'] > 0:
        L_consistency = exposure_consistency_loss(
            exposure[visible_idx],
            gaussians._exp_hist[visible_idx],
            gaussians._exp_hist_count[visible_idx],
        )

    loss = (
        lambda_weights['rgb'] * L_rgb +
        lambda_weights['smooth'] * L_smooth +
        lambda_weights['range'] * L_range +
        lambda_weights['consistency'] * L_consistency
    )

    return loss, {
        'L_rgb': L_rgb.item(),
        'L_smooth': L_smooth.item(),
        'L_range': L_range.item(),
        'L_consistency': L_consistency.item()
    }
