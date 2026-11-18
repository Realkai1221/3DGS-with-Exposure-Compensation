from .exposure_losses import (
    exposure_rgb_loss,
    exposure_smooth_loss,
    exposure_range_loss,
    exposure_consistency_loss,
)
from .exposure_adjustment import (
    adjust_color_with_exposure,
    adjust_depth_with_exposure,
)

def init_exposure_supervision(
    rendered_image, ground_truth_image,
    exposure_map,
    exposure_tensor,
    neighbor_idx,
    multiview_exposure_dict,
    lambda_weights=None
):
    if lambda_weights is None:
        lambda_weights = {
            'rgb': 1.0,
            'smooth': 0.1,
            'range': 0.01,
            'consistency': 0.1
        }

    L_rgb = exposure_rgb_loss(rendered_image, ground_truth_image, exposure_map)
    L_smooth = exposure_smooth_loss(exposure_tensor, neighbor_idx)
    L_range = exposure_range_loss(exposure_tensor)
    L_consistency = exposure_consistency_loss(exposure_tensor, multiview_exposure_dict)

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
