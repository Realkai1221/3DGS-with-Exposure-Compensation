"""Gaussian model that carries one exposure factor per Gaussian.

In v1/v2 the exposure factors lived in a separate ``ExposureModel`` that was
resized by hand after every densification step.  New Gaussians were given a
constant (0.95 + noise) and surviving values were copied by index, which no
longer lines up once Gaussians have been pruned.

Here the exposure factor is an ordinary parameter of the Gaussian model, in the
same optimizer as position, colour and opacity.  The stock 3DGS densification
code then prunes it for us, and the small overrides below make clones and split
children copy their parent's value exactly.

Written against the original (2023) ``scene/gaussian_model.py``.
"""

import torch
from torch import nn

from scene.gaussian_model import GaussianModel

from .neighbors import compute_knn

SH_C0 = 0.28209479177387814
EXPOSURE_MIN = 0.05
EXPOSURE_MAX = 1.5


class ExposureGaussianModel(GaussianModel):

    def __init__(self, sh_degree, exposure_init_value=0.95, history_window=10):
        super().__init__(sh_degree)
        self.exposure_init_value = float(exposure_init_value)
        self.exposure_lr = 0.0
        self.history_window = int(history_window)

        self._exposure = torch.empty(0)
        # Ring buffer with the last `history_window` exposure values each
        # Gaussian had when it was visible (used by the consistency loss).
        self._exp_hist = torch.empty(0)
        self._exp_hist_count = torch.empty(0, dtype=torch.long)
        # [N, k] indices of each Gaussian's nearest neighbours (smoothness loss).
        self.knn_idx = None

        self._pending_parents = None
        self._pending_new_exposure = None
        self._exposure_initialised = False

    # ------------------------------------------------------------------ access

    @property
    def get_exposure(self):
        return torch.clamp(self._exposure, EXPOSURE_MIN, EXPOSURE_MAX)

    def _ensure_exposure(self):
        n = self._xyz.shape[0]
        if self._exposure.shape[0] != n:
            self._exposure = nn.Parameter(torch.full(
                (n,), self.exposure_init_value, dtype=torch.float, device=self._xyz.device))

    def _reset_history(self):
        n = self._xyz.shape[0]
        device = self._xyz.device
        self._exp_hist = torch.zeros((n, self.history_window), device=device)
        self._exp_hist_count = torch.zeros((n,), dtype=torch.long, device=device)

    # ---------------------------------------------------------- initialisation

    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        self._ensure_exposure()

    def load_ply(self, *args, **kwargs):
        super().load_ply(*args, **kwargs)
        self._ensure_exposure()

    @torch.no_grad()
    def initialise_exposure(self, values=None):
        """Set the starting exposure of every Gaussian (call once, before training).

        The SfM colour of each point is what the camera saw, i.e. it already
        includes the exposure.  Dividing the starting colour by the exposure keeps
        colour x exposure equal to the SfM colour, so training starts from the
        same picture as plain 3DGS.
        """
        if self._exposure_initialised:
            raise RuntimeError("initialise_exposure() must only be called once")
        self._ensure_exposure()
        if values is None:
            values = torch.full_like(self._exposure, self.exposure_init_value)
        values = values.to(self._exposure.device, self._exposure.dtype)
        values = values.clamp(EXPOSURE_MIN, EXPOSURE_MAX)
        if values.shape != self._exposure.shape:
            raise ValueError("expected %s exposure values, got %s"
                             % (tuple(self._exposure.shape), tuple(values.shape)))
        self._exposure.data.copy_(values)
        rgb = 0.5 + SH_C0 * self._features_dc.data
        self._features_dc.data.copy_((rgb / values.view(-1, 1, 1) - 0.5) / SH_C0)
        self._exposure_initialised = True

    def training_setup(self, training_args):
        super().training_setup(training_args)
        self._ensure_exposure()
        self.optimizer.add_param_group(
            {'params': [self._exposure], 'lr': self.exposure_lr, 'name': 'exposure'})
        self._reset_history()

    def set_exposure_lr(self, lr):
        self.exposure_lr = lr
        for group in self.optimizer.param_groups:
            if group["name"] == "exposure":
                group["lr"] = lr

    # ------------------------------------------------------------ densification

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Same selection as the stock method; needed so we know who the parents are.
        mask = torch.norm(grads, dim=-1) >= grad_threshold
        mask = torch.logical_and(
            mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        self._pending_parents = torch.nonzero(mask).squeeze(1)
        try:
            super().densify_and_clone(grads, grad_threshold, scene_extent)
        finally:
            self._pending_parents = None

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points,), device=self._xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        mask = padded_grad >= grad_threshold
        mask = torch.logical_and(
            mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        self._pending_parents = torch.nonzero(mask).squeeze(1)
        try:
            super().densify_and_split(grads, grad_threshold, scene_extent, N)
        finally:
            self._pending_parents = None

    def _resolve_parents(self, n_new, postfix_args):
        """Parent index of each of the `n_new` Gaussians about to be appended."""
        selected = self._pending_parents
        if selected is None:
            raise RuntimeError(
                "densification_postfix() was called outside densify_and_clone/"
                "densify_and_split, so the parents of the new Gaussians are unknown.")
        if selected.numel() == 0:
            if n_new != 0:
                raise RuntimeError("exposure inheritance: %d new Gaussians but no parents" % n_new)
            return selected
        if n_new % selected.numel() != 0:
            raise RuntimeError(
                "exposure inheritance: %d new Gaussians for %d selected parents. The "
                "densification rule in scene/gaussian_model.py differs from the 2023 "
                "code this class was written for." % (n_new, selected.numel()))
        parents = selected.repeat(n_new // selected.numel())
        # Both clone and split copy the parent's rotation unchanged, which gives an
        # exact check that `parents` really matches the rows being appended.
        for arg in postfix_args:
            if torch.is_tensor(arg) and arg.shape == (n_new, 4):
                if not torch.equal(arg, self._rotation[parents]):
                    raise RuntimeError(
                        "exposure inheritance: new Gaussians do not match the selected "
                        "parents. The densification rule in scene/gaussian_model.py "
                        "differs from the 2023 code this class was written for.")
                break
        return parents

    def densification_postfix(self, new_xyz, *args, **kwargs):
        parents = self._resolve_parents(new_xyz.shape[0], args)
        # Children start with exactly their parent's exposure.
        self._pending_new_exposure = self._exposure.detach()[parents].clone()
        try:
            super().densification_postfix(new_xyz, *args, **kwargs)
        finally:
            self._pending_new_exposure = None

        self._exp_hist = torch.cat((self._exp_hist, self._exp_hist[parents]), dim=0)
        self._exp_hist_count = torch.cat((self._exp_hist_count, self._exp_hist_count[parents]), dim=0)
        if self.knn_idx is not None:
            # Until the next full rebuild a child uses its parent and the parent's
            # neighbours as its own neighbourhood.
            child_rows = torch.cat(
                (parents[:, None], self.knn_idx[parents][:, :-1]), dim=1)
            self.knn_idx = torch.cat((self.knn_idx, child_rows), dim=0)

    def cat_tensors_to_optimizer(self, tensors_dict):
        if "exposure" not in tensors_dict:
            if self._pending_new_exposure is None:
                raise RuntimeError("no exposure values prepared for the new Gaussians")
            tensors_dict = dict(tensors_dict)
            tensors_dict["exposure"] = self._pending_new_exposure
        optimizable_tensors = super().cat_tensors_to_optimizer(tensors_dict)
        self._exposure = optimizable_tensors["exposure"]
        return optimizable_tensors

    def prune_points(self, mask):
        valid = ~mask
        # The stock _prune_optimizer() loops over every parameter group, so the
        # exposure group is pruned together with the other attributes.
        super().prune_points(mask)
        for group in self.optimizer.param_groups:
            if group["name"] == "exposure":
                self._exposure = group["params"][0]
        self._exp_hist = self._exp_hist[valid]
        self._exp_hist_count = self._exp_hist_count[valid]
        if self.knn_idx is not None:
            new_index = torch.cumsum(valid.long(), dim=0) - 1
            idx = self.knn_idx[valid]
            neighbour_kept = valid[idx]
            own_row = torch.arange(idx.shape[0], device=idx.device)[:, None].expand_as(idx)
            # A neighbour that was pruned is replaced by the Gaussian itself,
            # which contributes zero to the smoothness loss.
            self.knn_idx = torch.where(neighbour_kept, new_index[idx], own_row)

    # ------------------------------------------------------ neighbours/history

    @torch.no_grad()
    def rebuild_neighbors(self, k=8):
        self.knn_idx = compute_knn(self.get_xyz, k)

    @torch.no_grad()
    def update_exposure_history(self, visible_idx):
        """Record the current exposure of the Gaussians seen in this view."""
        if visible_idx.numel() == 0:
            return
        slot = self._exp_hist_count[visible_idx] % self.history_window
        self._exp_hist[visible_idx, slot] = self.get_exposure[visible_idx]
        self._exp_hist_count[visible_idx] += 1

    # ------------------------------------------------------------------ saving

    @torch.no_grad()
    def save_exposure(self, path, iteration):
        torch.save({
            'exposure_values': self.get_exposure.detach().cpu(),
            'exposure_raw': self._exposure.detach().cpu(),
            'iteration': iteration,
        }, path)

    @torch.no_grad()
    def load_exposure(self, path):
        data = torch.load(path, map_location="cpu")
        self.set_exposure_tensor(data['exposure_raw'])

    @torch.no_grad()
    def set_exposure_tensor(self, raw):
        """Replace the exposure parameter (used when loading a model or checkpoint)."""
        device = self._xyz.device if self._xyz.numel() > 0 else raw.device
        self._exposure = nn.Parameter(raw.detach().clone().float().to(device))
        self._exposure_initialised = True

    def capture_exposure(self):
        """Exposure state for a training checkpoint (the stock capture() does not
        know about it): the raw parameter and the consistency history."""
        return {
            'exposure': self._exposure.detach().clone(),
            'exp_hist': self._exp_hist.clone(),
            'exp_hist_count': self._exp_hist_count.clone(),
        }

    @torch.no_grad()
    def restore_exposure_history(self, state):
        """Put the consistency history back after restore(), which starts it empty.
        Returns False (and leaves it empty) if the checkpoint holds no matching history."""
        hist, count = state.get('exp_hist'), state.get('exp_hist_count')
        n = self._xyz.shape[0]
        if hist is None or count is None or \
                tuple(hist.shape) != (n, self.history_window) or tuple(count.shape) != (n,):
            return False
        self._exp_hist = hist.to(self._xyz.device).clone()
        self._exp_hist_count = count.to(self._xyz.device).clone()
        return True

    @torch.no_grad()
    def save_baked_ply(self, path):
        """Save a .ply whose colours already include each Gaussian's exposure.

        Standard viewers (e.g. SIBR) know nothing about the exposure map and would
        show the exposure-free colours.  Multiplying every Gaussian's colour by its
        own exposure is a close approximation of render x exposure map that they
        can display.  Use render_exposure.py for the exact images.
        """
        e = self.get_exposure.view(-1, 1, 1)
        dc = self._features_dc.data.clone()
        rest = self._features_rest.data.clone()
        try:
            self._features_dc.data.copy_(((0.5 + SH_C0 * dc) * e - 0.5) / SH_C0)
            self._features_rest.data.copy_(rest * e)
            self.save_ply(path)
        finally:
            self._features_dc.data.copy_(dc)
            self._features_rest.data.copy_(rest)
