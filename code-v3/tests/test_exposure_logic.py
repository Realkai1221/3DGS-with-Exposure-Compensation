"""Checks for the exposure code that need neither a GPU nor the 3DGS repository.

    python tests/test_exposure_logic.py

Each test targets one of the problems found in v1/v2.  The rasteriser and the
Gaussian model are replaced by the CPU stand-ins in fake_3dgs.py, so this checks
the logic (bookkeeping, gradients), not the picture quality.
"""

import io
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import fake_3dgs  # noqa: E402

fake_3dgs.install()

from exposure.exposure_factors import initialize_exposure_from_projection  # noqa: E402
from exposure.exposure_gaussians import ExposureGaussianModel, SH_C0  # noqa: E402
from exposure.exposure_init_call import init_exposure_supervision  # noqa: E402
from exposure.exposure_losses import (  # noqa: E402
    exposure_consistency_loss, exposure_range_loss, exposure_smooth_loss)
from exposure.exposure_render import render_with_exposure  # noqa: E402
from exposure.neighbors import compute_knn  # noqa: E402

OPT = SimpleNamespace(percent_dense=0.01)
CAM = SimpleNamespace(image_height=16, image_width=16)
BG = torch.zeros(3)


def make_model(n, seed=0, setup=True):
    torch.manual_seed(seed)
    g = ExposureGaussianModel(3)
    xyz = torch.rand(n, 3) * 1.6 - 0.8
    colors = torch.rand(n, 3) * 0.8 + 0.1
    g.create_from_pcd(xyz, colors)
    if setup:
        g.exposure_lr = 0.001
        g.training_setup(OPT)
    return g, colors


def all_params(g):
    return [g._xyz, g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation, g._exposure]


def tag_gaussians(g):
    """Give every Gaussian an id (kept in a colour channel, which stock 3DGS copies
    to children) and an exposure that is a known function of that id."""
    n = g._xyz.shape[0]
    ids = torch.arange(n, dtype=torch.float)
    g._features_dc.data[:, 0, 0] = ids
    g._exposure.data.copy_(0.1 + ids * 0.002)


def expected_exposure(g):
    return 0.1 + g._features_dc.data[:, 0, 0] * 0.002


def test_1_projection_initialisation():
    g, colors = make_model(400, setup=False)
    g._xyz.data[0] = torch.tensor([5.0, 0.0, 0.0])          # outside every image
    W = 64
    x = (2 * torch.arange(W) + 1) / W - 1                    # pixel-centre x in NDC
    image = ((x + 1) / 2).view(1, 1, W).expand(3, W, W).contiguous()
    cam = SimpleNamespace(original_image=image, full_proj_transform=torch.eye(4))
    e_init, seen = initialize_exposure_from_projection(g.get_xyz, [cam, cam], default=0.95)

    assert not seen[0] and e_init[0] == 0.95
    assert seen[1:].all()
    want = ((g.get_xyz[1:, 0] + 1) / 2).clamp(0.1, 1.0)
    assert torch.allclose(e_init[1:], want, atol=1e-4), (e_init[1:] - want).abs().max()
    assert e_init[1:].std() > 0.1, "exposures should differ between Gaussians"

    # colour x exposure must still equal the SfM colour after initialisation
    g.initialise_exposure(e_init)
    shown = (0.5 + SH_C0 * g._features_dc.data[:, 0, :]) * g.get_exposure[:, None]
    assert torch.allclose(shown, colors, atol=1e-5)


def test_2_children_inherit_and_stay_aligned():
    g, _ = make_model(300)
    # one optimizer step so that Adam has state for every group
    (sum(p.sum() for p in all_params(g)) * 0.0).backward()
    g.optimizer.step()
    g.optimizer.zero_grad(set_to_none=True)
    tag_gaussians(g)
    g.rebuild_neighbors(4)
    g.update_exposure_history(torch.arange(300))

    total_new = 0
    for round_ in range(3):
        n_before = g._xyz.shape[0]
        g.xyz_gradient_accum = torch.rand(n_before, 1)
        g.denom = torch.ones(n_before, 1)
        g.densify_and_prune(0.5, 0.3, 10.0, None)            # clone + split + prune
        n = g._xyz.shape[0]
        total_new += abs(n - n_before)

        assert g._exposure.shape == (n,)
        assert torch.allclose(g._exposure.data, expected_exposure(g), atol=1e-6), \
            "an exposure no longer belongs to its Gaussian"
        assert g._exp_hist.shape == (n, g.history_window) and g._exp_hist_count.shape == (n,)
        assert torch.allclose(g._exp_hist[:, 0], expected_exposure(g), atol=1e-6)
        assert g.knn_idx.shape == (n, 4) and g.knn_idx.min() >= 0 and g.knn_idx.max() < n

        group = [grp for grp in g.optimizer.param_groups if grp["name"] == "exposure"][0]
        assert group["params"][0] is g._exposure
        for grp in g.optimizer.param_groups:
            state = g.optimizer.state.get(grp["params"][0])
            if state is not None:
                assert state["exp_avg"].shape == grp["params"][0].shape

        (sum(p.sum() for p in all_params(g)) * 0.0).backward()
        g.optimizer.step()
        g.optimizer.zero_grad(set_to_none=True)
    assert total_new > 0, "the test never changed the number of Gaussians"


def test_3_consistency_loss_has_gradient():
    g, _ = make_model(60)
    idx = torch.arange(60)
    zero = exposure_consistency_loss(g.get_exposure[idx], g._exp_hist[idx], g._exp_hist_count[idx])
    assert zero.item() == 0.0                                 # nothing remembered yet

    g._exposure.data.copy_(torch.rand(60) * 0.4 + 0.3)
    before = g.get_exposure.detach().clone()
    g.update_exposure_history(idx)
    g._exposure.data.add_(torch.randn(60) * 0.03)
    cur = g.get_exposure
    loss = exposure_consistency_loss(cur[idx], g._exp_hist[idx], g._exp_hist_count[idx])
    assert torch.allclose(loss, ((cur - before) ** 2 / 4).mean(), atol=1e-7)
    grad, = torch.autograd.grad(loss, g._exposure)
    assert (grad != 0).all()
    # gradient descent moves each exposure back towards the remembered value
    assert ((cur.detach() - before) * grad > 0).all()


def test_4_smoothness_runs_on_large_scenes():
    torch.manual_seed(0)
    pts = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [7.0, 0, 0], [8.0, 0, 0]])
    knn = compute_knn(pts, 2)
    assert knn.tolist() == [[1, 2], [0, 2], [1, 0], [4, 2], [3, 2]] or \
        [sorted(r) for r in knn.tolist()] == [[1, 2], [0, 2], [0, 1], [2, 4], [2, 3]]
    dup = torch.cat((pts, pts))                               # clones sit on their parents
    assert compute_knn(dup, 3).shape == (10, 3)

    n = 20000                                                 # v2 returned 0 above 10,000
    pts = torch.rand(n, 3)
    knn = compute_knn(pts, 8)
    assert knn.shape == (n, 8)
    assert (knn != torch.arange(n)[:, None]).all()
    e = torch.rand(n, requires_grad=True)
    loss = exposure_smooth_loss(e, knn)
    assert loss.item() > 0
    grad, = torch.autograd.grad(loss, e)
    assert (grad != 0).float().mean() > 0.99

    sample = torch.randint(0, n, (500,))
    grad, = torch.autograd.grad(exposure_smooth_loss(e, knn, sample), e)
    touched = torch.zeros(n, dtype=torch.bool)
    touched[sample] = True
    touched[knn[sample].reshape(-1)] = True
    assert (grad[~touched] == 0).all() and (grad[touched] != 0).any()


def test_5_each_exposure_reaches_the_image():
    g, _ = make_model(40)
    g._exposure.data.fill_(1.0)
    pkg = render_with_exposure(CAM, g, None, BG, fake_3dgs.render)
    assert pkg["exposure_map"].shape == (1, 16, 16)
    assert torch.allclose(pkg["exposure_map"], torch.ones(1, 16, 16), atol=1e-5)
    assert torch.allclose(pkg["image"], pkg["render"] * pkg["exposure_map"])

    g._exposure.data.copy_(torch.rand(40) * 0.8 + 0.2)
    pkg = render_with_exposure(CAM, g, None, BG, fake_3dgs.render)
    assert pkg["exposure_map"].std() > 1e-3
    grads = torch.autograd.grad(pkg["exposure_map"].sum(),
                                [g._exposure, g._xyz, g._opacity, g._scaling],
                                allow_unused=True, retain_graph=True)
    assert grads[0].std() > 0, "every Gaussian should influence the map differently"
    assert (grads[0] > 0).all()
    assert all(x is None for x in grads[1:]), "the exposure pass must not train geometry"

    # the colour image still trains geometry, and its loss reaches the exposures
    target = torch.rand(3, 16, 16)
    grads = torch.autograd.grad(((pkg["image"] - target) ** 2).mean(), [g._exposure, g._xyz])
    assert (grads[0] != 0).all() and (grads[1] != 0).any()

    off = render_with_exposure(CAM, g, None, BG, fake_3dgs.render, use_exposure=False)
    assert torch.equal(off["image"], off["render"])


def test_6_range_loss_can_pull_back():
    g, _ = make_model(20)
    g._exposure.data[0] = 3.0                                 # beyond the clamp at 1.5
    g._exposure.data[1] = -1.0
    grad, = torch.autograd.grad(exposure_range_loss(g._exposure), g._exposure)
    assert grad[0] > 0 and grad[1] < 0


def test_7_training_iterations_run():
    g, _ = make_model(120)
    g.initialise_exposure(torch.rand(120) * 0.8 + 0.2)
    weights = {'rgb': 1.0, 'smooth': 0.1, 'range': 0.01, 'consistency': 0.05}
    target = torch.rand(3, 16, 16)
    g.rebuild_neighbors(4)
    for it in range(1, 7):
        pkg = render_with_exposure(CAM, g, None, BG, fake_3dgs.render)
        visible_idx = torch.nonzero(pkg["visibility_filter"]).squeeze(1)
        loss, parts = init_exposure_supervision(pkg["image"], target, g, visible_idx, weights, 50)
        loss = loss + (pkg["image"] - target).abs().mean()
        assert torch.isfinite(loss)
        loss.backward()
        with torch.no_grad():
            g.update_exposure_history(visible_idx)
            if it > 1:
                assert parts['L_consistency'] >= 0 and parts['L_smooth'] > 0
            if it % 2 == 0:
                n = g._xyz.shape[0]
                g.xyz_gradient_accum = torch.rand(n, 1)
                g.denom = torch.ones(n, 1)
                g.densify_and_prune(0.5, 0.2, 10.0, None)
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
        assert g._exposure.shape[0] == g._xyz.shape[0] == g.knn_idx.shape[0]


def _train_step(g, target, weights, seed):
    torch.manual_seed(seed)                                   # the smoothness loss samples
    pkg = render_with_exposure(CAM, g, None, BG, fake_3dgs.render)
    visible_idx = torch.nonzero(pkg["visibility_filter"]).squeeze(1)
    loss, _ = init_exposure_supervision(pkg["image"], target, g, visible_idx, weights, 50)
    loss = loss + (pkg["image"] - target).abs().mean()
    loss.backward()
    with torch.no_grad():
        g.update_exposure_history(visible_idx)
        g.optimizer.step()
        g.optimizer.zero_grad(set_to_none=True)


def test_8_checkpoint_resume_continues_identically():
    weights = {'rgb': 1.0, 'smooth': 0.1, 'range': 0.01, 'consistency': 0.05}
    target = torch.rand(3, 16, 16)
    a, _ = make_model(150)
    a.initialise_exposure(torch.rand(150) * 0.8 + 0.2)
    a.rebuild_neighbors(4)
    for it in range(4):
        _train_step(a, target, weights, it)
        if it == 1:                                           # change the number of Gaussians
            with torch.no_grad():
                n = a._xyz.shape[0]
                a.xyz_gradient_accum = torch.rand(n, 1)
                a.denom = torch.ones(n, 1)
                a.densify_and_prune(0.5, 0.2, 10.0, None)
    n = a._xyz.shape[0]
    assert n != 150 and a._exp_hist_count.max() > 0

    # what trainv3.save_checkpoint writes, through a real save / load
    state = a.capture_exposure()
    state['gaussians'] = a.capture()
    state['iteration'] = 4
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)

    # what trainv3 does with --start_checkpoint: a fresh model from the SfM points first
    b, _ = make_model(90, seed=5, setup=False)
    b.exposure_lr = 0.001
    b.set_exposure_tensor(loaded['exposure'])
    b.restore(loaded['gaussians'], OPT)
    assert b.restore_exposure_history(loaded)
    assert b._xyz.shape[0] == n and b._exposure.shape == (n,)
    group = [grp for grp in b.optimizer.param_groups if grp["name"] == "exposure"][0]
    assert group["params"][0] is b._exposure
    assert b.optimizer.state[b._exposure]["exp_avg"].shape == (n,)
    assert torch.equal(b._exp_hist, a._exp_hist) and torch.equal(b._exp_hist_count, a._exp_hist_count)
    assert b.knn_idx is None                                  # trainv3 rebuilds it when missing

    a.rebuild_neighbors(4)
    b.rebuild_neighbors(4)
    for it in range(4, 8):
        _train_step(a, target, weights, it)
        _train_step(b, target, weights, it)
    for pa, pb in zip(all_params(a), all_params(b)):
        assert torch.allclose(pa, pb, atol=1e-6), (pa - pb).abs().max()
    assert torch.equal(a._exp_hist_count, b._exp_hist_count)
    assert torch.allclose(a._exp_hist, b._exp_hist, atol=1e-6)

    # a checkpoint without a matching history is refused, not mis-assigned
    assert not b.restore_exposure_history({'exposure': loaded['exposure']})
    wrong = {'exp_hist': loaded['exp_hist'][:-1], 'exp_hist_count': loaded['exp_hist_count'][:-1]}
    assert not b.restore_exposure_history(wrong)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", type(exc).__name__, exc)
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
