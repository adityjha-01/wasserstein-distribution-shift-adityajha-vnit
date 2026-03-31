import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wasserstein_distance, kurtosis, skew

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
BATCH_SIZE   = 128
DATA_ROOT    = "./data"
SUBSAMPLE    = 10000      # max points for W-distance (speed)
OUTPUT_PATH  = "distribution_analysis.png"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_SEED   = 42

# ─────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────
def set_seed(seed: int = TORCH_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

# ─────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────
def load_single_batch(batch_size: int = BATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Download MNIST (if needed) and return one flattened batch."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset   = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=True, transform=transform, download=True
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    images, labels = next(iter(loader))
    return images.view(images.size(0), -1).to(DEVICE), labels.to(DEVICE)

# ─────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────
class SimpleNN(nn.Module):
    """
    Three-layer MLP with BatchNorm and Dropout.
    Exposes all intermediate activations for analysis.

    Returns: (z1, a1, z2, a2, z3_logits)
        z1  – pre-activation  layer 1
        a1  – post-ReLU       layer 1
        z2  – pre-activation  layer 2
        a2  – post-ReLU       layer 2
        out – final logits
    """
    def __init__(self,
                 input_dim  : int = 784,
                 hidden1    : int = 256,
                 hidden2    : int = 64,
                 num_classes: int = 10,
                 dropout    : float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden1)
        self.bn1  = nn.BatchNorm1d(hidden1)
        self.fc2  = nn.Linear(hidden1, hidden2)
        self.bn2  = nn.BatchNorm1d(hidden2)
        self.fc3  = nn.Linear(hidden2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        z1  = self.fc1(x)
        a1  = F.relu(self.bn1(z1))
        a1  = self.drop(a1)
        z2  = self.fc2(a1)
        a2  = F.relu(self.bn2(z2))
        out = self.fc3(a2)
        return z1, a1, z2, a2, out

# ─────────────────────────────────────────
# 4. Forward Pass
# ─────────────────────────────────────────
def run_forward(model: nn.Module,
                images: torch.Tensor) -> tuple:
    """Inference-only forward pass (no gradients)."""
    model.eval()
    with torch.no_grad():
        return model(images)

# ─────────────────────────────────────────
# 5. Statistics
# ─────────────────────────────────────────
def layer_stats(arr: np.ndarray) -> dict:
    """Descriptive statistics for a flattened layer array."""
    return {
        "mean"    : float(np.mean(arr)),
        "std"     : float(np.std(arr)),
        "skew"    : float(skew(arr)),
        "kurtosis": float(kurtosis(arr)),
        "sparsity": float((arr == 0).mean()),   # fraction of exact zeros
    }


def compute_wasserstein(tensors: dict[str, np.ndarray],
                        max_pts: int = SUBSAMPLE) -> dict[str, float]:
    """W-distance between every consecutive pair of layer distributions."""
    names  = list(tensors.keys())
    arrays = list(tensors.values())
    rng    = np.random.default_rng(42)

    distances: dict[str, float] = {}
    for i in range(len(arrays) - 1):
        a, b = arrays[i], arrays[i + 1]
          # if len(a) > max_pts:
          #  a = rng.choice(a, size=max_pts, replace=False)
          #  b = rng.choice(b, size=max_pts, replace=False)
        size = min(len(a), len(b), max_pts)
        a = rng.choice(a, size=size, replace=False)
        b = rng.choice(b, size=size, replace=False)
        key = f"{names[i]} → {names[i + 1]}"
        distances[key] = wasserstein_distance(a, b)

    return distances


def dead_neuron_ratio(a1: torch.Tensor) -> float:
    """
    Fraction of neurons that output 0 for every sample in the batch.
    High ratio → dying ReLU problem.
    """
    dead = (a1 == 0).all(dim=0).sum().item()
    return dead / a1.shape[1]


def top_k_predictions(logits: torch.Tensor,
                      labels: torch.Tensor,
                      k: int = 3) -> float:
    """Top-k accuracy on the single batch (random weights → ~chance level)."""
    probs   = torch.softmax(logits, dim=1)
    topk    = probs.topk(k, dim=1).indices
    correct = topk.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct

# ─────────────────────────────────────────
# 6. Print Summary
# ─────────────────────────────────────────
def print_summary(layer_data   : dict[str, np.ndarray],
                  distances    : dict[str, float],
                  dead_ratio   : float,
                  topk_acc     : float) -> None:

    bar = "─" * 52
    print(f"\n{bar}")
    print("  Neural Network Activation Analysis")
    print(bar)

    print("\n📊 Layer Statistics:")
    header = f"  {'Layer':<12}{'Mean':>8}{'Std':>8}{'Skew':>8}{'Kurt':>8}{'Sparse':>8}"
    print(header)
    print("  " + "─" * 48)
    for name, arr in layer_data.items():
        s = layer_stats(arr)
        print(f"  {name:<12}"
              f"{s['mean']:>8.3f}"
              f"{s['std']:>8.3f}"
              f"{s['skew']:>8.3f}"
              f"{s['kurtosis']:>8.3f}"
              f"{s['sparsity']:>8.2%}")

    print("\n📐 Wasserstein Distances:")
    for k, v in distances.items():
        print(f"  {k:<30}: {v:.6f}")

    print(f"\n💀 Dead neuron ratio (Layer 1): {dead_ratio:.2%}")
    print(f"🎯 Top-3 accuracy (random init): {topk_acc:.2%}")
    print(f"\n{bar}\n")

# ─────────────────────────────────────────
# 7. Visualisation  (4-panel dashboard)
# ─────────────────────────────────────────
def plot_dashboard(layer_data: dict[str, np.ndarray],
                   distances : dict[str, float],
                   stats_map : dict[str, dict]) -> None:

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467BD"]
    fig    = plt.figure(figsize=(18, 13))
    fig.suptitle("Activation Distribution Dashboard  ·  SimpleNN on MNIST",
                 fontsize=15, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── Panel A: overlapping histograms ──────────────────────────────
    ax_hist = fig.add_subplot(gs[0, :2])
    for (name, arr), color in zip(layer_data.items(), colors):
        ax_hist.hist(arr, bins=100, alpha=0.50,
                     label=name, color=color, density=True)
    ax_hist.set_title("A · Distribution Shift Across Layers", fontweight="bold")
    ax_hist.set_xlabel("Activation value")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)

    # ── Panel B: Wasserstein bar chart ───────────────────────────────
    ax_wd = fig.add_subplot(gs[0, 2])
    wd_keys = list(distances.keys())
    wd_vals = list(distances.values())
    bars = ax_wd.barh(wd_keys, wd_vals,
                      color=colors[:len(wd_keys)], edgecolor="white")
    for bar, val in zip(bars, wd_vals):
        ax_wd.text(bar.get_width() + max(wd_vals) * 0.02,
                   bar.get_y() + bar.get_height() / 2,
                   f"{val:.4f}", va="center", fontsize=8)
    ax_wd.set_title("B · Wasserstein Distances", fontweight="bold")
    ax_wd.set_xlabel("Distance")
    ax_wd.invert_yaxis()

    # ── Panel C: per-layer box plots ─────────────────────────────────
    ax_box = fig.add_subplot(gs[1, :2])
    names  = list(layer_data.keys())
    data   = [layer_data[n] for n in names]
    bp     = ax_box.boxplot(data, labels=names, patch_artist=True,
                            notch=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax_box.set_title("C · Per-layer Box Plots  (outliers hidden)", fontweight="bold")
    ax_box.set_ylabel("Activation value")

    # ── Panel D: sparsity bar chart ──────────────────────────────────
    ax_sp = fig.add_subplot(gs[1, 2])
    sparsities = [stats_map[n]["sparsity"] for n in names]
    ax_sp.bar(names, sparsities, color=colors[:len(names)], edgecolor="white")
    ax_sp.set_title("D · Sparsity  (fraction of zeros)", fontweight="bold")
    ax_sp.set_ylabel("Sparsity ratio")
    ax_sp.set_ylim(0, 1)
    for i, v in enumerate(sparsities):
        ax_sp.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    # ── Panel E: KDE curves (smooth density) ─────────────────────────
    ax_kde = fig.add_subplot(gs[2, :2])
    from scipy.stats import gaussian_kde
    x_range = np.linspace(-5, 5, 500)
    for (name, arr), color in zip(layer_data.items(), colors):
        sample = arr if len(arr) <= 20_000 else \
                 np.random.default_rng(0).choice(arr, 20_000, replace=False)
        try:
            kde = gaussian_kde(sample)
            ax_kde.plot(x_range, kde(x_range), label=name,
                        color=color, linewidth=1.8)
            ax_kde.fill_between(x_range, kde(x_range), alpha=0.12, color=color)
        except np.linalg.LinAlgError:
            pass    # singular KDE (all-zero layer) – skip silently
    ax_kde.set_title("E · KDE  (Kernel Density Estimate)", fontweight="bold")
    ax_kde.set_xlabel("Activation value")
    ax_kde.set_ylabel("Density")
    ax_kde.set_xlim(-5, 5)
    ax_kde.legend(fontsize=8)

    # ── Panel F: stats table ─────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 2])
    ax_tbl.axis("off")
    ax_tbl.set_title("F · Layer Statistics", fontweight="bold")
    col_labels = ["Layer", "Mean", "Std", "Skew", "Sparse"]
    rows = []
    for name in names:
        s = stats_map[name]
        rows.append([name,
                     f"{s['mean']:.3f}",
                     f"{s['std']:.3f}",
                     f"{s['skew']:.3f}",
                     f"{s['sparsity']:.1%}"])
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved → {OUTPUT_PATH}")
    plt.show()

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main() -> None:
    set_seed()

    # 1. Data
    images, labels = load_single_batch()

    # 2. Model + forward pass
    model = SimpleNN().to(DEVICE)
    z1, a1, z2, a2, out = run_forward(model, images)

    # 3. Flatten to numpy
    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.cpu().numpy().flatten()

    layer_data: dict[str, np.ndarray] = {
        "Input"    : to_np(images),
        "Layer 1"  : to_np(z1),
        "ReLU 1"   : to_np(a1),
        "Layer 2"  : to_np(z2),
        "ReLU 2"   : to_np(a2),
    }

    # 4. Metrics
    distances  = compute_wasserstein(layer_data)
    stats_map  = {name: layer_stats(arr) for name, arr in layer_data.items()}
    dead_ratio = dead_neuron_ratio(a1)
    topk_acc   = top_k_predictions(out, labels, k=3)

    # 5. Print + plot
    print_summary(layer_data, distances, dead_ratio, topk_acc)
    plot_dashboard(layer_data, distances, stats_map)


if __name__ == "__main__":
    main()