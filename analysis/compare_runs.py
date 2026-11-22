"""Compare the Colab runs image by image.

usage: python compare_runs.py <folder with the results> <output folder>

Each run is either a folder <name>/ (as Safari unzips it) or results_<name>.zip
in the results folder.  The script

* checks that every run was tested on the same photos, in the same order;
* prints an overview and a per-image table against plain 3DGS;
* scores every run again on the right halves of the test photos, the only part
  that 3dgs2024_exp is scored on (that procedure trains on the left halves);
* writes worst_views_<run>.png, right_halves_<run>.png and per_image_psnr.csv
  into the output folder.
"""
import csv, glob, json, os, re, sys, zipfile

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


def read(path):  # RGB, values 0..1, channels first
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


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


def right_half(img):  # the crop render.py of the 2024 code makes with --train_test_exp
    return img[..., img.shape[-1] // 2:]


def log_value(text, pattern):
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None


def stem(filename):
    return os.path.splitext(os.path.basename(filename))[0]


runs, plain_gt = {}, {}
for name in NAMES:
    root = locate(name)
    if root is None:
        continue
    pv = json.load(open(os.path.join(root, "per_view.json")))
    key = next(iter(pv))
    test = os.path.join(root, "test", key)
    views = sorted(pv[key]["PSNR"])
    r = {"root": root, "test": test, "views": views, "per_view": pv[key]}
    path = os.path.join(root, "run_info.json")
    r["info"] = json.load(open(path)) if os.path.isfile(path) else {}
    # cameras.json lists the test cameras first, in the order of the test images
    path = os.path.join(root, "cameras.json")
    r["photos"] = [stem(c["img_name"]) for c in json.load(open(path))[:len(views)]] if os.path.isfile(path) else []
    path = os.path.join(root, "train_log.txt")
    log = open(path, errors="replace").read() if os.path.isfile(path) else ""
    r["test7k"] = log_value(log, r"\[ITER 7000\] Evaluating test: L1 \S+ PSNR (\S+)")
    r["test30k"] = log_value(log, r"\[ITER 30000\] Evaluating test: L1 \S+ PSNR (\S+)")
    r["train30k"] = log_value(log, r"\[ITER 30000\] Evaluating train: L1 \S+ PSNR (\S+)")
    r["init_line"] = next((l for l in log.splitlines() if "Exposure initialised" in l), "")
    r["resume_line"] = next((l for l in log.splitlines() if "Resumed from" in l), "")

    # The 3x4 colour transform per photo that the 2024 code learns (identity when it is not used):
    # rendered pixel (row vector) @ M[:, :3] + M[:, 3]
    path = os.path.join(root, "exposure.json")
    affine = {stem(k): np.array(m, dtype=np.float64) for k, m in json.load(open(path)).items()} if os.path.isfile(path) else None

    free_dir = os.path.join(test, "renders_exposure_free")
    map_dir = os.path.join(test, "exposure_maps")
    for k in ("free_psnr", "free_ssim", "bright", "map_mean", "map_std", "rh_psnr", "rh_ssim", "undone_psnr", "undone_ssim"):
        r[k] = {}
    r["check"], r["gt_diff"], r["half"] = [], 0.0, False
    for i, v in enumerate(views):
        gt = read(os.path.join(test, "gt", v))
        ren = read(os.path.join(test, "renders", v))
        r["check"].append(abs(psnr(ren, gt) - r["per_view"]["PSNR"][v]))
        r["bright"][v] = float(ren.mean() / gt.mean())
        if name == "plain":
            plain_gt[v] = gt
        ref = plain_gt.get(v)
        if ref is not None and gt.shape != ref.shape:
            r["half"], ref = True, right_half(ref)
        if ref is None or gt.shape != ref.shape:
            r["gt_diff"] = float("inf")
        else:
            r["gt_diff"] = max(r["gt_diff"], float((gt - ref).abs().max()))
        rg, rr = (gt, ren) if r["half"] else (right_half(gt), right_half(ren))
        r["rh_psnr"][v], r["rh_ssim"][v] = psnr(rr, rg), ssim(rr, rg)
        if r["half"] and affine and i < len(r["photos"]) and r["photos"][i] in affine:
            # the same render with the photo's learned colour transform taken out again
            m = torch.tensor(affine[r["photos"][i]], dtype=torch.float32)
            raw = ((ren.permute(1, 2, 0) - m[:, 3]) @ torch.linalg.inv(m[:, :3])).permute(2, 0, 1).clamp(0, 1)
            r["undone_psnr"][v], r["undone_ssim"][v] = psnr(raw, gt), ssim(raw, gt)
        if os.path.isfile(os.path.join(free_dir, v)):
            free = read(os.path.join(free_dir, v))
            r["free_psnr"][v], r["free_ssim"][v] = psnr(free, gt), ssim(free, gt)
        if os.path.isfile(os.path.join(map_dir, v)):
            e = read(os.path.join(map_dir, v))[0]
            r["map_mean"][v], r["map_std"][v] = float(e.mean()), float(e.std())

    e = None
    pts = glob.glob(os.path.join(root, "point_cloud", "iteration_*", "exposures.pt"))       # code-v3
    if pts:
        e = torch.load(pts[0], map_location="cpu")["exposure_values"].float()
    pts = glob.glob(os.path.join(root, "point_cloud", "iteration_*", "exposure_model.pt"))  # v1
    if pts:
        e = torch.load(pts[0], map_location="cpu")["exposure"].float().clamp(0.05, 1.5)  # v1 clamps them like this
    if e is not None:
        q = torch.quantile(e[torch.randperm(e.numel())[:1_000_000]], torch.tensor([0.01, 0.5, 0.99]))
        r["exposures"] = (e.numel(), float(e.mean()), float(e.std()), *map(float, q))
    if affine:
        ident = np.hstack([np.eye(3), np.zeros((3, 1))])
        gain = {k: float(np.mean(0.5 * m[:, :3].sum(axis=0) + m[:, 3]) / 0.5) for k, m in affine.items()}
        test_gain = [gain[p] for p in r["photos"] if p in gain]
        r["affine"] = (len(affine), max(float(np.abs(m - ident).max()) for m in affine.values()),
                       np.array(list(gain.values())), np.array(test_gain))
    runs[name] = r
    print(f"read {name}: {len(views)} test images{' (right halves)' if r['half'] else ''}, "
          f"own PSNR matches metrics.py within {max(r['check']):.4f} dB")

mean = lambda d: float(np.mean(list(d.values()))) if d else float("nan")

print("\n=== Same test photos in every run?")
for name, r in runs.items():
    same_names = r["photos"] == runs["plain"]["photos"] if "plain" in runs else None
    r["aligned"] = r["gt_diff"] <= 1.5 / 255 and same_names is not False
    print(f"  {name:16} photo names {'same' if same_names else 'DIFFERENT' if same_names is False else '?'}, "
          f"largest pixel difference of the photos {r['gt_diff'] * 255:.1f}/255 -> {'OK' if r['aligned'] else 'NOT COMPARABLE'}")

print("\n=== Overview (test images; 'free' = render before the exposure map is applied)")
print(f"{'run':16}{'part':>6}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'freePSNR':>10}{'freeSSIM':>10}{'bright':>8}{'mapStd':>8}{'@7k':>7}{'@30k':>7}{'train':>7}{'Gauss':>7}{'min':>5}  resumed")
for name, r in runs.items():
    pv, info = r["per_view"], r["info"]
    print(f"{name:16}{'right' if r['half'] else 'full':>6}{mean(pv['PSNR']):8.2f}{mean(pv['SSIM']):8.4f}{mean(pv['LPIPS']):8.4f}"
          f"{mean(r['free_psnr']):10.2f}{mean(r['free_ssim']):10.4f}{mean(r['bright']):8.3f}{mean(r['map_std']):8.3f}"
          f"{r['test7k'] or float('nan'):7.2f}{r['test30k'] or float('nan'):7.2f}{r['train30k'] or float('nan'):7.2f}"
          f"{info.get('gaussians', 0) / 1e6:6.2f}M{info.get('train_minutes', 0):5.0f}  {info.get('resumed_from_iteration', '?')}")
for name, r in runs.items():
    if r["init_line"] or r["resume_line"]:
        print(f"  {name}: {r['init_line'].strip()} {r['resume_line'].strip()}")
    if "exposures" in r:
        n, m, s, q1, q50, q99 = r["exposures"]
        print(f"  {name}: final exposures of {n:,} Gaussians: mean {m:.3f}, std {s:.3f}, 1% {q1:.3f}, median {q50:.3f}, 99% {q99:.3f}")
    if "affine" in r:
        n, dev, g, tg = r["affine"]
        print(f"  {name}: colour transforms of {n} photos, largest difference from 'no change' {dev:.4f}; brightness factor "
              f"all photos {g.mean():.3f} +/- {g.std():.3f} [{g.min():.3f}, {g.max():.3f}]"
              + (f", test photos {tg.mean():.3f} +/- {tg.std():.3f} [{tg.min():.3f}, {tg.max():.3f}]" if len(tg) else ""))

if "plain" not in runs:
    sys.exit()
base = runs["plain"]["per_view"]["PSNR"]
order = sorted(base, key=base.get)
worst = order[:8]
full = [n for n in runs if n != "plain" and not runs[n]["half"] and runs[n]["aligned"]]

print("\n=== PSNR per test image, worst plain images first (other runs: difference to plain; f = before the exposure map)")
print(f"{'image':10}{'plain':>7}" + "".join(f"{n[:14]:>15}{'f':>6}" for n in full))
for v in order:
    row = f"{v:10}{base[v]:7.2f}"
    for n in full:
        r = runs[n]
        row += f"{r['per_view']['PSNR'].get(v, float('nan')) - base[v]:+15.2f}{r['free_psnr'].get(v, float('nan')) - base[v]:+6.2f}"
    print(row)


def change_line(n, d, what):
    better = sum(x > 0 for x in d.values())
    return (f"{n}: mean change {np.mean(list(d.values())):+.2f} dB against {what}; on plain's 8 worst images "
            f"{np.mean([d[v] for v in worst if v in d]):+.2f}, on the other {len(d) - 8} {np.mean([d[v] for v in order[8:] if v in d]):+.2f}; "
            f"better on {better}/{len(d)} images")


for n in full:
    r = runs[n]
    d = {v: r["per_view"]["PSNR"][v] - base[v] for v in base if v in r["per_view"]["PSNR"]}
    line = change_line(n, d, "plain")
    if r["map_std"]:
        x = np.array([r["map_std"][v] for v in d])
        line += f"; correlation of the change with the exposure-map spread {np.corrcoef(x, np.array(list(d.values())))[0, 1]:+.2f}"
    print(line)

print("\n=== Right halves of the test photos (the part 3dgs2024_exp is scored on; LPIPS not recomputed)")
print(f"{'run':16}{'PSNR':>8}{'SSIM':>8}   full-image PSNR")
for name, r in runs.items():
    if r["aligned"]:
        print(f"{name:16}{mean(r['rh_psnr']):8.2f}{mean(r['rh_ssim']):8.4f}   "
              + ("-" if r["half"] else f"{mean(r['per_view']['PSNR']):.2f}"))
        if r["undone_psnr"]:
            print(f"{'  same, colour transform taken out':40}{mean(r['undone_psnr']):8.2f}{mean(r['undone_ssim']):8.4f}")
if "3dgs2024" in runs:
    ref = runs["3dgs2024"]["rh_psnr"]
    for n in [n for n in runs if runs[n]["half"] and runs[n]["aligned"]]:
        print(change_line(n, {v: runs[n]["rh_psnr"][v] - ref[v] for v in ref if v in runs[n]["rh_psnr"]}, "3dgs2024 on right halves"))
        if runs[n]["undone_psnr"]:
            print(change_line(n + " without its colour transforms",
                              {v: runs[n]["undone_psnr"][v] - ref[v] for v in ref if v in runs[n]["undone_psnr"]}, "3dgs2024 on right halves"))

with open(os.path.join(out, "per_image_psnr.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "photo"] + [f"{n} full" for n in runs if not runs[n]["half"]] + [f"{n} right half" for n in runs])
    for i, v in enumerate(runs["plain"]["views"]):
        photo = runs["plain"]["photos"][i] if i < len(runs["plain"]["photos"]) else ""
        w.writerow([v, photo] + [f"{runs[n]['per_view']['PSNR'].get(v, float('nan')):.3f}" for n in runs if not runs[n]["half"]]
                   + [f"{runs[n]['rh_psnr'].get(v, float('nan')):.3f}" for n in runs])
print("\nwrote", os.path.join(out, "per_image_psnr.csv"))


def figure(path, rows_spec):
    """rows_spec: list of (caption, [image paths], crop to right half?)"""
    rows = []
    for caption, paths, crop in rows_spec:
        tiles = []
        for p in paths:
            t = cv2.imread(p)
            if crop and t.shape[1] > min(cv2.imread(q).shape[1] for q in paths):
                t = t[:, t.shape[1] // 2:]
            tiles.append(t)
        h = 480
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h), interpolation=cv2.INTER_AREA) for t in tiles]
        row = np.concatenate(tiles, axis=1)
        strip = np.full((34, row.shape[1], 3), 255, np.uint8)
        cv2.putText(strip, caption, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.concatenate((strip, row), axis=0))
    width = max(x.shape[1] for x in rows)
    rows = [np.pad(x, ((0, 0), (0, width - x.shape[1]), (0, 0)), constant_values=255) for x in rows]
    cv2.imwrite(path, np.concatenate(rows, axis=0))
    print("wrote", path)


# The images that lost most against plain, for each full-image run.
P = runs["plain"]["test"]
for n in full:
    r = runs[n]
    d = {v: r["per_view"]["PSNR"][v] - base[v] for v in base if v in r["per_view"]["PSNR"]}
    spec = []
    for v in sorted(d, key=d.get)[:5]:
        paths = [os.path.join(P, "gt", v), os.path.join(P, "renders", v), os.path.join(r["test"], "renders", v)]
        if os.path.isfile(os.path.join(r["test"], "exposure_maps", v)):
            paths.append(os.path.join(r["test"], "exposure_maps", v))
        spec.append((f"{v}: plain {base[v]:.1f} dB, {n} {r['per_view']['PSNR'][v]:.1f} dB", paths, False))
    figure(os.path.join(out, f"worst_views_{n}.png"), spec)
print("  (columns: photo | plain | this run | its exposure map, if it has one)")

# Right halves: the largest changes of each right-half run against 3dgs2024.
if "3dgs2024" in runs:
    R = runs["3dgs2024"]
    for n in [n for n in runs if runs[n]["half"] and runs[n]["aligned"]]:
        r = runs[n]
        d = {v: r["rh_psnr"][v] - R["rh_psnr"][v] for v in R["rh_psnr"] if v in r["rh_psnr"]}
        spec = [(f"{v}: 3dgs2024 {R['rh_psnr'][v]:.1f} dB, {n} {r['rh_psnr'][v]:.1f} dB (right halves)",
                 [os.path.join(r["test"], "gt", v), os.path.join(R["test"], "renders", v), os.path.join(r["test"], "renders", v)], True)
                for v in sorted(d, key=lambda v: -abs(d[v]))[:6]]
        figure(os.path.join(out, f"right_halves_{n}.png"), spec)
    print("  (columns: right half of the photo | 3dgs2024 | this run)")
