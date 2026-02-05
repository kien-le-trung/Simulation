import matplotlib.pyplot as plt
import numpy as np

# heatmaps
xlabels = ["shortest", "short", "medium", "long", "longest"]  # time bins
ylabels = ["closest", "close", "medium", "far", "farthest"]   # distance bins
N_DIRECTIONS = 8

def plot_heatmap(mat, title, xlabels, ylabels, annotate=True):
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(xlabels)), xlabels)
    plt.yticks(range(len(ylabels)), ylabels)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat.dtype == int:
                    txt = str(mat[i, j])
                else:
                    txt = f"{mat[i, j]:.2f}"
                plt.text(j, i, txt, ha="center", va="center", fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def phit_tensor_from_hist(
    hist,
    d_min=0.10,
    d_max=0.80,
    t_min=1.0,
    t_max=7.0,
    n_d=5,
    n_t=5,
    n_dir=8,
):
    def _level5(x, xmin, xmax):
        u = (x - xmin) / (xmax - xmin + 1e-12)
        if u < 0.2: return 0
        if u < 0.4: return 1
        if u < 0.6: return 2
        if u < 0.8: return 3
        return 4

    hits = np.zeros((n_d, n_t, n_dir), dtype=float)
    total = np.zeros((n_d, n_t, n_dir), dtype=float)

    for d, t, direction, hit in zip(
        hist.get("d", []),
        hist.get("t", []),
        hist.get("direction", []),
        hist.get("hit", []),
    ):
        i = _level5(float(d), d_min, d_max)
        j = _level5(float(t), t_min, t_max)
        k = int(np.clip(direction, 0, n_dir - 1))
        total[i, j, k] += 1.0
        hits[i, j, k] += 1.0 if hit else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        phit = np.divide(hits, total, out=np.zeros_like(hits), where=total > 0)
    return phit


def plot_phit_d_dir(phit_tensor, t_idx=None, reduce="mean"):
    if t_idx is None:
        if reduce == "mean":
            mat = np.mean(phit_tensor, axis=1)
        elif reduce == "min":
            mat = np.min(phit_tensor, axis=1)
        elif reduce == "max":
            mat = np.max(phit_tensor, axis=1)
        else:
            raise ValueError("reduce must be one of: mean, min, max")
        title = f"P(hit) vs (d,dir), t={reduce}"
    else:
        j = int(np.clip(t_idx, 0, 4))
        mat = phit_tensor[:, j, :]
        title = f"P(hit) vs (d,dir), t_idx={j}"

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xticks(range(N_DIRECTIONS), [f"dir{d}" for d in range(N_DIRECTIONS)])
    plt.yticks(range(5), ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.show()