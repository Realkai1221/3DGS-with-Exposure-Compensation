# 3DGS with Exposure Compensation

Code, training notebook, results and LaTeX source for the paper *How Effective Can
Exposure Compensation Reduce Floating Artifacts in 3D Gaussian Splatting for Indoor
Scene Reconstruction?*

| Folder | Contents |
|---|---|
| `code-v3/` | The exposure-compensation code. It is copied into the official gaussian-splatting repository (commit `2eee0e2`); see `code-v3/README.md`. Includes a self-test and the v1 baseline. |
| `colab/exposure_3dgs_colab.ipynb` | Notebook that trains and scores every method on a Colab GPU. |
| `colab/<run>/` | One folder per finished run: metrics per test image (`per_view.json`), averages (`results.json`), settings (`run_info.json`, `cfg_args*`), training log, test renders, photographs, exposure maps and renders before the exposure map is applied. |
| `analysis/compare_runs.py` | Compares the runs image by image. |
| `exposure-3dgs-latex-v2/` | LaTeX source of the paper (compiled on Overleaf). |

Not in this repository, because they are too large for GitHub: the dataset (`room.zip`,
324 photos taken from a video of one room, kept in the Google Drive folder `3dgs`) and the
source videos.

The notebook reads the code from `code-v3.zip` in the Google Drive folder `3dgs`. To make
that zip after changing the code, run this in the repository folder:

```bash
zip -r colab/code-v3.zip code-v3 -x "*/__pycache__/*" "*.pyc" "*.DS_Store"
```

To compare the runs (needs numpy, opencv-python and torch):

```bash
python analysis/compare_runs.py colab analysis/output
```
