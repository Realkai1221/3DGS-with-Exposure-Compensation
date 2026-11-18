"""k-nearest-neighbour graph over the Gaussian centres (for the smoothness loss)."""

import torch


def _knn_scipy(points_np, k):
    from scipy.spatial import cKDTree
    tree = cKDTree(points_np)
    _, indices = tree.query(points_np, k=k, workers=-1)
    return indices


def _knn_sklearn(points_np, k):
    from sklearn.neighbors import KDTree
    tree = KDTree(points_np, leaf_size=40)
    _, indices = tree.query(points_np, k=k)
    return indices


def _knn_faiss_gpu(points_np, k):
    import faiss
    if faiss.get_num_gpus() == 0:
        raise RuntimeError("faiss has no GPU")
    index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(3))
    index.add(points_np)
    _, indices = index.search(points_np, k)
    return indices


def _knn_torch(points, k):
    # Brute force; only sensible for small point sets.
    out = []
    for start in range(0, points.shape[0], 2048):
        d = torch.cdist(points[start:start + 2048], points)
        out.append(d.topk(k, dim=1, largest=False).indices)
    return torch.cat(out, dim=0)


@torch.no_grad()
def compute_knn(points, k=8, backend="auto"):
    """Indices [N, k] of the k nearest other points of every point.

    backend: "auto" tries faiss (GPU) -> scipy -> scikit-learn -> torch.
    """
    points = points.detach()
    n = points.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return None
    q = k + 1  # the query also returns the point itself

    order = ["faiss", "scipy", "sklearn", "torch"] if backend == "auto" else [backend]
    indices = None
    errors = []
    for name in order:
        try:
            if name == "torch":
                if backend == "auto" and n > 50000:
                    raise RuntimeError("too many points for the brute-force fallback")
                indices = _knn_torch(points, q)
            else:
                points_np = points.cpu().numpy().astype("float32")
                fn = {"faiss": _knn_faiss_gpu, "scipy": _knn_scipy, "sklearn": _knn_sklearn}[name]
                indices = torch.from_numpy(fn(points_np, q))
            break
        except Exception as exc:  # try the next backend
            errors.append("%s: %s" % (name, exc))
    if indices is None:
        raise RuntimeError("no k-NN backend worked (install scipy): " + "; ".join(errors))

    indices = indices.long().to(points.device)
    # Drop the point itself from its own row.  With duplicated points (fresh
    # clones sit exactly on their parent) "itself" is not always the first hit.
    rows = torch.arange(n, device=points.device)[:, None]
    is_self = indices == rows
    no_self = ~is_self.any(dim=1)
    is_self[no_self, -1] = True
    return indices[~is_self].view(n, k)
