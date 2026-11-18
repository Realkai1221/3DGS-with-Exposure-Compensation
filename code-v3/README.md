# Exposure compensation for 3DGS, version 3

Same method as `trainv2.py` (one exposure factor per Gaussian, four exposure
losses on top of the 3DGS loss).  This version repairs the places where the v2
code did not do what the paper describes.

**Status: runs end to end, full-length results pending.**  `python tests/test_exposure_logic.py`
passes 8/8 on CPU, using stand-ins for the Gaussian model and the rasteriser.
The code has also been run with the real CUDA rasteriser inside the
gaussian-splatting repository (commit 2eee0e2) on a Colab T4: training,
rendering and metrics complete.  Only short runs (3,100 iterations) of the
exposure method exist so far, so there is no full-length number for it yet.
`colab/exposure_3dgs_colab.ipynb` trains the baseline and the exposure method
at full length.  A run can be continued from a checkpoint
(`--checkpoint_iterations`, `--checkpoint_copy_dir`, `--start_checkpoint`); the
continue step is covered by the self-test but has not been run on a GPU yet.

`baselines/` holds the original v1 code, which the notebook trains for
comparison together with the official 2024 version of 3DGS (see
`baselines/README.md`).  Neither comparison run has been made yet.

The model, renderer, camera and metrics code it relies on were checked by reading
the gaussian-splatting source at commit `2eee0e2` (1 Nov 2023, the last commit of
2023).  `colab/exposure_3dgs_colab.ipynb` installs exactly that commit.

## What was wrong and what changed

| # | Problem in v2 | Fix in v3 | Where |
|---|---|---|---|
| 1 | `initialize_exposure_from_projection` was never called; every exposure started at 0.95 | Each Gaussian starts from the mean grey level it projects to in the training images. Its starting colour is divided by that value, so colour x exposure still equals the SfM colour | `exposure/exposure_factors.py`, `ExposureGaussianModel.initialise_exposure` |
| 2 | New Gaussians got 0.95 + noise, and old values were copied by index, which misaligns them after pruning | Exposure is a parameter of the Gaussian model, in the same optimizer as position and colour. Stock pruning keeps it aligned; clones and split children copy their parent's value exactly | `exposure/exposure_gaussians.py` |
| 3 | The consistency loss was built from Python floats and had no gradient | Tensor version: variance over the last views that saw each Gaussian plus its current value, with the gradient through the current value | `exposure_consistency_loss` |
| 4 | The smoothness loss returned 0 above 10,000 Gaussians and ran every 2,000 iterations | Runs every iteration on a sample of the visible Gaussians at any scene size. The neighbour graph follows densification and is rebuilt every `--knn_interval` iterations | `exposure_smooth_loss`, `exposure/neighbors.py` |
| 5 | The exposure map was four fixed patterns scaled by the mean and spread of the visible exposures | The map is rasterised from the per-Gaussian exposures with the same alpha-compositing weights as the colour image and divided by the coverage, giving the average exposure of the Gaussians covering each pixel (second render pass using `override_color`; no CUDA changes) | `exposure/exposure_render.py` |
| 6 | Densification threshold x5 was hard-coded, so its effect could not be separated from the exposure losses | Stock threshold by default; `--densify_scale 5` reproduces v2 | `trainv3.py` |

Three more differences from stock 3DGS turned up while fixing these. Each is now
stock by default, with a flag to get the v2 behaviour back:

* **Opacity reset was missing** in v1 and v2. Stock 3DGS resets opacity every
  3,000 iterations; that is its main way of removing floaters. `--no_opacity_reset`
* Position gradients were clipped to norm 1. `--clip_xyz_grad`
* Densification statistics were scaled by the mean exposure. Removed; the
  exposure map now scales the gradients itself.

Two details that differ from the v2 code on purpose:

* **Eq. 1 is applied as `C = C_adj * e`.** The exposure-free render is multiplied
  by the map and compared with the untouched photo. Dividing the photo by the map
  lets the optimiser lower the loss by raising the exposure (both sides shrink).
* **Evaluation uses render x exposure map.** `trainv3.py` reports PSNR on that
  image, and `render_exposure.py` writes it. The stock `render.py` would write the
  exposure-free colours, which are not comparable with the photos.

`AdaptiveTrainingConfig`, the 100-Gaussian sampling and the per-stage timers are
gone: they existed to work around the Python loops, which no longer exist.

## Install

Copy `trainv3.py`, `render_exposure.py` and the `exposure/` folder into the root
of the 2023 gaussian-splatting repository (next to `train.py`). The neighbour
search needs `scipy` (or scikit-learn, or faiss-gpu).

## Run

    python trainv3.py -s <dataset> -m output/room_v3 --eval
    python render_exposure.py -m output/room_v3
    python metrics.py -m output/room_v3

Each saved iteration contains `point_cloud.ply` (exposure-free colours),
`exposures.pt`, and `point_cloud_baked.ply` (colours multiplied by each
Gaussian's exposure, for viewers such as SIBR that cannot apply the map).

## Ablations

Change one thing at a time against the full method:

    # stock 3DGS through this script (sanity check: should match train.py)
    python trainv3.py -s <dataset> -m output/abl_plain --eval --no_exposure

    # full method
    python trainv3.py -s <dataset> -m output/abl_full --eval

    # is it the smoothness loss or the x5 threshold that cleans the wardrobe?
    python trainv3.py -s <dataset> -m output/abl_no_smooth --eval --lambda_smooth 0
    python trainv3.py -s <dataset> -m output/abl_dens5 --eval --densify_scale 5
    python trainv3.py -s <dataset> -m output/abl_plain_dens5 --eval --no_exposure --densify_scale 5

    # the other pieces
    python trainv3.py -s <dataset> -m output/abl_no_cons --eval --lambda_consistency 0
    python trainv3.py -s <dataset> -m output/abl_const_init --eval --exposure_init constant
    python trainv3.py -s <dataset> -m output/abl_no_reset --eval --no_opacity_reset

    # closest to v2's settings
    python trainv3.py -s <dataset> -m output/abl_like_v2 --eval --exposure_init constant \
        --densify_scale 5 --clip_xyz_grad --no_opacity_reset

## Limits

* One exposure per Gaussian is the same for every photo, so it cannot represent
  the camera changing exposure between photos.
* The second render pass makes each iteration slower than stock 3DGS.
* Projection initialisation does not test occlusion.
* Checkpoints from v1/v2 cannot be resumed.
