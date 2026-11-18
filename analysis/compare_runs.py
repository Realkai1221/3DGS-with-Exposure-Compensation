"""Compare the Colab runs image by image.

usage: python analysis/compare_runs.py colab <output folder>

Each run is either a folder colab/<name>/ (as Safari unzips it) or
colab/results_<name>.zip.  Prints an overview (including the scores of the
renders before the exposure map is applied), a per-image PSNR table against
plain 3DGS, and writes worst_views_<run>.png into the output folder: the five
test images that lost most against plain 3DGS, as photo | plain | run | map.

PSNR and SSIM are computed as in the gaussian-splatting code, and the script
checks that its PSNR matches the per_view.json written by metrics.py.
Needs numpy, opencv-python and torch.
"""
import glob, json, os, re, sys, zipfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F

NAMES = ["plain", "exposure_const", "exposure", "3dgs2024", "v1", "3dgs2024_exp"]
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def locate(name):
    folder = os.path.join(src, name)
    if os.path.isfile(os.path.join(folder, "per_view.json")):
        return folder
    z = os.path.join(src, f"results_{name}.zip")
    if os.path.isfile(z):
        dest = os.path.join(out, "runs")
        if not os.path.isfile(os.path.join(dest, name, "per_view.json")):
            with zipfile.ZipFile(z) as f:
                f.extractall(dest)
        return os.path.join(dest, name)
    return None


def read(path):
    return torch.from_numpy(cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0).permute(2, 0, 1)


def psnr(a, b):  # as utils/image_utils.py of 3DGS, one image
    return float(20 * torch.log10(1.0 / torch.sqrt(((a - b) ** 2).mean())))


_g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
_g = (_g / _g.sum())[:, None] @ (_g / _g.sum())[None, :]
WINDOW = _g.expand(3, 1, 11, 11).contiguous()


def ssim(a, b):  # as utils/loss_utils.py of 3DGS
    a, b = a[None], b[None]
    mu1, mu2 = F.conv2d(a, WINDOW, padding=5, groups=3), F.conv2d(b, WINDOW, padding=5, groups=3)
    s11 = F.conv2d(a * a, WINDOW, padding=5, groups=3) - mu1 ** 2
    s22 = F.conv2d(b * b, WINDOW, padding=5, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, WINDOW, padding=5, groups=3) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def log_value(text, pattern):
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None


runs = {}
for name in NAMES:
    root = locate(name)
    if root is None:
        continue
    pv = json.load(open(os.path.join(root, "per_view.json")))
    key = next(iter(pv))
    test = os.path.join(root, "test", key)
    views = sorted(pv[key]["PSNR"])
    r = {"root": root, "test": test, "views": views, "per_view": pv[key]}
    r["info"] = json.load(open(os.path.join(root, "run_info.json"))) if os.path.isfile(os.path.join(root, "run_info.json")) else {}
    log = open(os.path.join(root, "train_log.txt"), errors="replace").read() if os.path.isfile(os.path.join(root, "train_log.txt")) else ""
    r["test7k"] = log_value(log, r"\[ITER 7000\] Evaluating test: L1 \S+ PSNR (\S+)")
    r["test30k"] = log_value(log, r"\[ITER 30000\] Evaluating test: L1 \S+ PSNR (\S+)")
    r["train30k"] = log_value(log, r"\[ITER 30000\] Evaluating train: L1 \S+ PSNR (\S+)")
    r["init_line"] = next((l for l in log.splitlines() if "Exposure initialised" in l), "")
    r["resume_line"] = next((l for l in log.splitlines() if "Resumed from" in l), "")

    free_dir = os.path.join(test, "renders_exposure_free")
    map_dir = os.path.join(test, "exposure_maps")
    r["free_psnr"], r["free_ssim"], r["bright"], r["map_mean"], r["map_std"], r["check"] = {}, {}, {}, {}, {}, []
    for v in views:
        gt = read(os.path.join(test, "gt", v))
        ren = read(os.path.join(test, "renders", v))
        r["check"].append(abs(psnr(ren, gt) - r["per_view"]["PSNR"][v]))
        r["bright"][v] = float(ren.mean() / gt.mean())
        if os.path.isfile(os.path.join(free_dir, v)):
            free = read(os.path.join(free_dir, v))
            r["free_psnr"][v], r["free_ssim"][v] = psnr(free, gt), ssim(free, gt)
        if os.path.isfile(os.path.join(map_dir, v)):
            e = read(os.path.join(map_dir, v))[0]
            r["map_mean"][v], r["map_std"][v] = float(e.mean()), float(e.std())
    pts = glob.glob(os.path.join(root, "point_cloud", "iteration_*", "exposures.pt"))
    if pts:
        e = torch.load(pts[0], map_location="cpu")["exposure_values"].float()
        q = torch.quantile(e[torch.randperm(e.numel())[:1_000_000]], torch.tensor([0.01, 0.5, 0.99]))
        r["exposures"] = (e.numel(), float(e.mean()), float(e.std()), *map(float, q))
    runs[name] = r
    print(f"read {name}: {len(views)} test images (own PSNR matches metrics.py within {max(r['check']):.4f} dB)")

mean = lambda d: float(np.mean(list(d.values()))) if d else float("nan")
print("\n=== Overview (test images; 'free' = render before the exposure map is applied)")
print(f"{'run':16}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'freePSNR':>10}{'freeSSIM':>10}{'bright':>8}{'mapStd':>8}{'@7k':>7}{'@30k':>7}{'train':>7}{'Gauss':>7}{'min':>5}  resumed")
for name, r in runs.items():
    pv, info = r["per_view"], r["info"]
    print(f"{name:16}{mean(pv['PSNR']):8.2f}{mean(pv['SSIM']):8.4f}{mean(pv['LPIPS']):8.4f}"
          f"{mean(r['free_psnr']):10.2f}{mean(r['free_ssim']):10.4f}{mean(r['bright']):8.3f}{mean(r['map_std']):8.3f}"
          f"{r['test7k'] or float('nan'):7.2f}{r['test30k'] or float('nan'):7.2f}{r['train30k'] or float('nan'):7.2f}"
          f"{info.get('gaussians', 0) / 1e6:6.2f}M{info.get('train_minutes', 0):5.0f}  {info.get('resumed_from_iteration', '?')}")
for name, r in runs.items():
    if r["init_line"] or r["resume_line"] or "exposures" in r:
        print(f"  {name}: {r['init_line'].strip()} {r['resume_line'].strip()}")
        if "exposures" in r:
            n, m, s, q1, q50, q99 = r["exposures"]
            print(f"  {name}: final exposures of {n:,} Gaussians: mean {m:.3f}, std {s:.3f}, 1% {q1:.3f}, median {q50:.3f}, 99% {q99:.3f}")

if "plain" not in runs:
    sys.exit()
base = runs["plain"]["per_view"]["PSNR"]
others = [n for n in runs if n != "plain"]
order = sorted(base, key=base.get)
print("\n=== PSNR per test image, worst plain images first (other runs: difference to plain; f = before the exposure map)")
print(f"{'image':10}{'plain':>7}" + "".join(f"{n[:14]:>15}{'f':>6}" for n in others))
for v in order:
    row = f"{v:10}{base[v]:7.2f}"
    for n in others:
        r = runs[n]
        d = r["per_view"]["PSNR"].get(v, float("nan")) - base[v]
        f = r["free_psnr"].get(v, float("nan")) - base[v]
        row += f"{d:+15.2f}{f:+6.2f}"
    print(row)
worst = order[:8]
for n in others:
    r = runs[n]
    d = {v: r["per_view"]["PSNR"][v] - base[v] for v in base if v in r["per_view"]["PSNR"]}
    better = sum(x > 0 for x in d.values())
    line = (f"{n}: mean change {np.mean(list(d.values())):+.2f} dB; on plain's 8 worst images {np.mean([d[v] for v in worst]):+.2f}, "
            f"on the other {len(d) - 8} {np.mean([d[v] for v in order[8:]]):+.2f}; better than plain on {better}/{len(d)} images")
    if r["map_std"]:
        x = np.array([r["map_std"][v] for v in d])
        y = np.array(list(d.values()))
        line += f"; correlation of the change with the exposure-map spread {np.corrcoef(x, y)[0, 1]:+.2f}"
    print(line)

# Pictures of the images that lost most against plain, for each other run.
for n in others:
    r = runs[n]
    d = {v: r["per_view"]["PSNR"][v] - base[v] for v in base if v in r["per_view"]["PSNR"]}
    rows = []
    for v in sorted(d, key=d.get)[:5]:
        tiles = [cv2.imread(os.path.join(runs["plain"]["test"], "gt", v)),
                 cv2.imread(os.path.join(runs["plain"]["test"], "renders", v)),
                 cv2.imread(os.path.join(r["test"], "renders", v))]
        m = os.path.join(r["test"], "exposure_maps", v)
        if os.path.isfile(m):
            tiles.append(cv2.imread(m))
        h = 480
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h), interpolation=cv2.INTER_AREA) for t in tiles]
        row = np.concatenate(tiles, axis=1)
        strip = np.full((34, row.shape[1], 3), 255, np.uint8)
        cv2.putText(strip, f"{v}: plain {base[v]:.1f} dB, {n} {r['per_view']['PSNR'][v]:.1f} dB", (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.concatenate((strip, row), axis=0))
    width = max(x.shape[1] for x in rows)
    rows = [np.pad(x, ((0, 0), (0, width - x.shape[1]), (0, 0))) for x in rows]
    path = os.path.join(out, f"worst_views_{n}.png")
    cv2.imwrite(path, np.concatenate(rows, axis=0))
    print("wrote", path, "(columns: photo | plain | this run | its exposure map)")
