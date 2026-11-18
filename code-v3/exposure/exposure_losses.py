"""The four exposure losses, written with tensors so that gradients flow.

What was wrong in v2:
  * smooth loss: returned 0 above 10,000 Gaussians and was only evaluated every
    2,000 iterations, through a Python loop over every Gaussian.
  * consistency loss: built from Python floats, so it had no gradient at all.
"""

import torch


def exposure_rgb_loss(image_with_exposure, I_gt):
    """MSE between the exposure-adjusted render and the photograph."""
    return torch.nn.functional.mse_loss(image_with_exposure, I_gt)


def exposure_smooth_loss(exposure_tensor, neighbor_idx, sample_idx=None):
    """Mean squared exposure difference between each Gaussian and its neighbours.

    exposure_tensor: [N] exposure of every Gaussian.
    neighbor_idx:    [N, k] long tensor from compute_knn().
    sample_idx:      optional [M] subset of Gaussians to evaluate this iteration.
    Works for any scene size; cost is M x k.
    """
    if neighbor_idx is None or neighbor_idx.numel() == 0:
        return exposure_tensor.sum() * 0.0
    if sample_idx is None:
        centre = exposure_tensor
        neighbours = exposure_tensor[neighbor_idx]
    else:
        if sample_idx.numel() == 0:
            return exposure_tensor.sum() * 0.0
        centre = exposure_tensor[sample_idx]
        neighbours = exposure_tensor[neighbor_idx[sample_idx]]
    return ((centre[:, None] - neighbours) ** 2).mean()


def exposure_range_loss(exposure_tensor, low=0.1, high=1.0):
    """Penalise exposures outside [low, high].

    Pass the raw (unclamped) parameter: a clamped value has zero gradient outside
    the clamp, so it could never be pulled back.
    """
    lower = torch.relu(low - exposure_tensor)
    upper = torch.relu(exposure_tensor - high)
    return (lower ** 2 + upper ** 2).mean()


def exposure_consistency_loss(current, history, history_count):
    """Variance of each Gaussian's exposure over the recent views that saw it.

    current:       [M] exposure of the Gaussians visible now (carries gradient).
    history:       [M, W] their exposure the last W times they were visible.
    history_count: [M] how many values have been written to each history row.

    Each Gaussian has a single exposure, so "across views" means: across the
    training views that most recently updated it.  The variance is taken over
    the stored values plus the current one, and the gradient flows through the
    current value, pulling it towards what the other views agreed on.
    """
    window = history.shape[1]
    n = history_count.clamp(max=window)
    has_history = n >= 1
    if not has_history.any():
        return current.sum() * 0.0

    cur = current[has_history]
    hist = history[has_history].detach()
    n = n[has_history]
    valid = (torch.arange(window, device=hist.device)[None, :] < n[:, None]).to(hist.dtype)
    m = (n + 1).to(hist.dtype)

    mean = ((hist * valid).sum(dim=1) + cur) / m
    var = (((hist - mean[:, None]) ** 2 * valid).sum(dim=1) + (cur - mean) ** 2) / m
    return var.mean()
