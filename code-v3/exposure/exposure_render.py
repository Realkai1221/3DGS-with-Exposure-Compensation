"""Exposure map rendered from the per-Gaussian exposure factors.

v2 built its "spatial exposure map" from four fixed image-space patterns scaled
by the mean and standard deviation of the visible exposures, so an individual
Gaussian's exposure never reached the picture.

Here the map is produced by the rasteriser itself.  The stock 3DGS renderer can
splat any per-Gaussian value in place of the colour (`override_color`), so a
second pass with colour = exposure gives, for every pixel,

    E(pixel) = sum_i  w_i(pixel) * e_i  /  sum_i  w_i(pixel)

with exactly the alpha-compositing weights w_i used for the colour image, i.e.
the average exposure of the Gaussians that cover the pixel.  Dividing by the
coverage matters where the Gaussians are semi-transparent: a pixel covered at
50% then still gets the exposure of what covers it, not a mix with "no exposure".  This
is the pixel-accurate map described in the paper, without changing the CUDA code.
"""

import torch


class _DetachedGeometry:
    """A Gaussian model whose geometry carries no gradient.

    The exposure pass must only train the exposure factors; it should not move,
    resize or fade Gaussians in order to explain brightness.
    """

    def __init__(self, pc):
        self._pc = pc
        self.active_sh_degree = pc.active_sh_degree
        self.max_sh_degree = pc.max_sh_degree

    @property
    def get_xyz(self):
        return self._pc.get_xyz.detach()

    @property
    def get_opacity(self):
        return self._pc.get_opacity.detach()

    @property
    def get_scaling(self):
        return self._pc.get_scaling.detach()

    @property
    def get_rotation(self):
        return self._pc.get_rotation.detach()

    @property
    def get_features(self):
        return self._pc.get_features.detach()

    def get_covariance(self, scaling_modifier=1):
        return self._pc.get_covariance(scaling_modifier).detach()


def render_exposure_map(viewpoint_cam, gaussians, pipe, render_fn):
    """Exposure map [1, H, W] for one view, differentiable w.r.t. the exposures."""
    exposure = gaussians.get_exposure
    # Channel 0 accumulates w_i * e_i and channel 1 accumulates w_i (the coverage).
    color = torch.stack((exposure, torch.ones_like(exposure), torch.zeros_like(exposure)), dim=1)
    black = torch.zeros(3, dtype=torch.float32, device=exposure.device)
    pkg = render_fn(viewpoint_cam, _DetachedGeometry(gaussians), pipe, black,
                    override_color=color)
    weighted, coverage = pkg["render"][0:1], pkg["render"][1:2]
    # Average exposure of the Gaussians covering each pixel.  Pixels that no
    # Gaussian covers get neutral exposure 1.
    covered = coverage > 1e-6
    return torch.where(covered, weighted / coverage.clamp_min(1e-6), torch.ones_like(weighted))


def render_with_exposure(viewpoint_cam, gaussians, pipe, background, render_fn=None,
                         use_exposure=True):
    """Stock render plus the exposure map.

    Returns the stock render package with two extra entries:
      "exposure_map": [1, H, W]
      "image":        exposure-free render x exposure map, the picture that is
                      compared with the photograph.
    "render" stays the exposure-free image.
    """
    if render_fn is None:
        from gaussian_renderer import render as render_fn
    pkg = render_fn(viewpoint_cam, gaussians, pipe, background)
    raw = pkg["render"]
    if use_exposure:
        exposure_map = render_exposure_map(viewpoint_cam, gaussians, pipe, render_fn)
    else:
        exposure_map = torch.ones_like(raw[0:1])
    pkg["exposure_map"] = exposure_map
    pkg["image"] = raw * exposure_map
    return pkg
