"""Question 2: Robust Line and Circle Fitting using Sequential RANSAC.

Generates noisy point set conforming to a line and a circle (Listing 1), performs:
(a) 2-point RANSAC line estimation with ||[a,b]|| = 1 constraint and Total Least Squares refitting.
(b) Subtraction of line consensus and 3-point RANSAC circle estimation with Nelder-Mead refinement.
(c) Unified visualization matching Figure 1 of the assignment brief (both sample and refined fits).
(d) Real circle-first RANSAC execution and analysis of order sensitivity.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_noisy_data(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Generate noisy point set conforming to a ground-truth line and circle (Listing 1)."""
    rng = np.random.default_rng(seed)
    N = 100
    half_n = N // 2

    # Ground truth circle: center = (2, 3), r = 10, noise s = r/16
    circle_center_gt = np.array([2.0, 3.0])
    circle_radius_gt = 10.0
    s_circ = circle_radius_gt / 16.0
    t = rng.uniform(0, 2 * np.pi, half_n)
    radial_noise = s_circ * rng.standard_normal(half_n)
    X_circ = np.column_stack((
        circle_center_gt[0] + (circle_radius_gt + radial_noise) * np.cos(t),
        circle_center_gt[1] + (circle_radius_gt + radial_noise) * np.sin(t),
    ))

    # Ground truth line: y = m*x + b => -1*x + 2, noise s = 1.0
    line_slope_gt, line_intercept_gt = -1.0, 2.0
    s_line = 1.0
    x_line = np.linspace(-12.0, 12.0, half_n)
    y_line = line_slope_gt * x_line + line_intercept_gt + s_line * rng.standard_normal(half_n)
    X_line = np.column_stack((x_line, y_line))

    X = np.vstack((X_circ, X_line))
    return X, circle_center_gt, circle_radius_gt, line_slope_gt, line_intercept_gt


def line_from_two_points(p1: np.ndarray, p2: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return unit-normal line parameters (a, b, d) such that a*x + b*y = d and a^2 + b^2 = 1."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return None
    normal = np.array([-direction[1], direction[0]]) / length
    # Canonical sign convention: ensure a >= 0 (or b >= 0 if a == 0)
    if normal[0] < 0 or (abs(normal[0]) < 1e-9 and normal[1] < 0):
        normal = -normal
    d = float(normal @ p1)
    return float(normal[0]), float(normal[1]), d


def circle_from_three_points(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Compute circumcircle (cx, cy, radius) passing through three non-collinear points."""
    p1, p2, p3 = np.asarray(points, dtype=float)
    matrix = 2.0 * np.vstack((p2 - p1, p3 - p1))
    rhs = np.array([p2 @ p2 - p1 @ p1, p3 @ p3 - p1 @ p1])
    if abs(np.linalg.det(matrix)) < 1e-8:
        return None
    center = np.linalg.solve(matrix, rhs)
    radius = float(np.linalg.norm(center - p1))
    if not np.isfinite(radius) or radius <= 0:
        return None
    return float(center[0]), float(center[1]), radius


def ransac_line(
    points: np.ndarray,
    threshold: float = 0.85,
    min_inliers: int = 35,
    iterations: int = 500,
    seed: int = 0,
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Estimate line using 2-point RANSAC and Total Least Squares (TLS) SVD refitting."""
    rng = np.random.default_rng(seed)
    best_sample_model, best_mask, best_sample = None, None, None

    for _ in range(iterations):
        sample = rng.choice(len(points), size=2, replace=False)
        model = line_from_two_points(points[sample[0]], points[sample[1]])
        if model is None:
            continue
        a, b, d = model
        errors = np.abs(points @ np.array([a, b]) - d)
        mask = errors < threshold
        if best_mask is None or mask.sum() > best_mask.sum():
            best_sample_model, best_mask, best_sample = model, mask, sample

    if best_mask is None or best_mask.sum() < min_inliers:
        raise RuntimeError("RANSAC could not find line with sufficient inliers.")

    # Total Least Squares refit on consensus set using SVD
    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    if normal[0] < 0 or (abs(normal[0]) < 1e-9 and normal[1] < 0):
        normal = -normal
    d = float(normal @ centroid)
    refined_model = (float(normal[0]), float(normal[1]), d)

    return refined_model, best_mask, best_sample, best_sample_model


def ransac_circle(
    points: np.ndarray,
    threshold: float = 0.85,
    min_inliers: int = 30,
    iterations: int = 500,
    seed: int = 0,
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Estimate circle using 3-point RANSAC and Nelder-Mead geometric refitting."""
    rng = np.random.default_rng(seed)
    best_sample_model, best_mask, best_sample = None, None, None

    for _ in range(iterations):
        sample = rng.choice(len(points), size=3, replace=False)
        model = circle_from_three_points(points[sample])
        if model is None:
            continue
        cx, cy, radius = model
        errors = np.abs(np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)
        mask = errors < threshold
        if best_mask is None or mask.sum() > best_mask.sum():
            best_sample_model, best_mask, best_sample = model, mask, sample

    if best_mask is None or best_mask.sum() < min_inliers:
        raise RuntimeError("RANSAC could not find circle with sufficient inliers.")

    # Refine using Nelder-Mead (scipy.optimize.minimize)
    inliers = points[best_mask]
    res = minimize(
        lambda p: np.sum((np.hypot(inliers[:, 0] - p[0], inliers[:, 1] - p[1]) - p[2]) ** 2),
        x0=np.asarray(best_sample_model),
        method="Nelder-Mead",
    )
    refined_model = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
    return refined_model, best_mask, best_sample, best_sample_model


def run_circle_first_ransac(
    points: np.ndarray,
    threshold: float = 0.85,
    iterations: int = 1000,
    seed: int = 0,
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Run circle RANSAC for real on full point cloud without subtracting line."""
    rng = np.random.default_rng(seed)
    best_sample_model, best_mask, best_sample = None, None, None

    for _ in range(iterations):
        sample = rng.choice(len(points), size=3, replace=False)
        model = circle_from_three_points(points[sample])
        if model is None:
            continue
        cx, cy, radius = model
        errors = np.abs(np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)
        mask = errors < threshold
        if best_mask is None or mask.sum() > best_mask.sum():
            best_sample_model, best_mask, best_sample = model, mask, sample

    inliers = points[best_mask]
    res = minimize(
        lambda p: np.sum((np.hypot(inliers[:, 0] - p[0], inliers[:, 1] - p[1]) - p[2]) ** 2),
        x0=np.asarray(best_sample_model),
        method="Nelder-Mead",
    )
    refined_model = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
    return refined_model, best_mask, best_sample, best_sample_model


def run_q2(output_dir: Path):
    """Execute Question 2 pipeline, generate plots, and report numerical comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X, center_gt, r_gt, m_gt, b_gt = generate_noisy_data(seed=0)

    # Ground truth parameters:
    # Line: y = -x + 2 => x + y = 2 => normal [1/sqrt(2), 1/sqrt(2)], d = 2/sqrt(2) = sqrt(2)
    n_gt = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    d_gt = 2.0 / np.sqrt(2)

    # 1. Line-first RANSAC:
    line_ref, line_mask, line_sample, line_samp_m = ransac_line(
        X, threshold=0.85, min_inliers=35, iterations=500
    )
    rem_indices = np.where(~line_mask)[0]
    remnant = X[rem_indices]

    # Remnant circle RANSAC:
    circ_ref, circ_local_mask, circ_sample_local, circ_samp_m = ransac_circle(
        remnant, threshold=0.85, min_inliers=30, iterations=500
    )
    circ_global_sample = rem_indices[circ_sample_local]
    circ_global_mask = np.zeros(len(X), dtype=bool)
    circ_global_mask[rem_indices[circ_local_mask]] = True

    # 2. Circle-first RANSAC (run for real on full X):
    cf_ref, cf_mask, cf_sample, cf_samp_m = run_circle_first_ransac(
        X, threshold=0.85, iterations=1000
    )

    # Numerical reporting:
    print("=" * 60)
    print("QUESTION 2 NUMERICAL RESULTS:")
    print("=" * 60)
    print(f"Ground Truth Line: normal = [{n_gt[0]:.4f}, {n_gt[1]:.4f}], d = {d_gt:.4f}")
    print(f"Sample Line:       normal = [{line_samp_m[0]:.4f}, {line_samp_m[1]:.4f}], d = {line_samp_m[2]:.4f}")
    print(f"Refined Line (TLS): normal = [{line_ref[0]:.4f}, {line_ref[1]:.4f}], d = {line_ref[2]:.4f}")
    print(f"Line Inliers: {line_mask.sum()} / 100 points ({line_mask[50:].sum()} line pts + {line_mask[:50].sum()} circle pts)")
    print("-" * 60)
    print(f"Ground Truth Circle: center = ({center_gt[0]:.3f}, {center_gt[1]:.3f}), radius = {r_gt:.3f}")
    print(f"Sample Circle:       center = ({circ_samp_m[0]:.3f}, {circ_samp_m[1]:.3f}), radius = {circ_samp_m[2]:.3f}")
    print(f"Refined Circle:      center = ({circ_ref[0]:.3f}, {circ_ref[1]:.3f}), radius = {circ_ref[2]:.3f}")
    print(f"Circle Inliers: {circ_global_mask.sum()} / {len(remnant)} remnant ({circ_global_mask[:50].sum()} circle pts + {circ_global_mask[50:].sum()} line pts)")
    print("-" * 60)
    print("Circle-First RANSAC (Full Point Cloud X):")
    print(f"Sample Model:  center = ({cf_samp_m[0]:.3f}, {cf_samp_m[1]:.3f}), radius = {cf_samp_m[2]:.3f}")
    print(f"Refined Model: center = ({cf_ref[0]:.3f}, {cf_ref[1]:.3f}), radius = {cf_ref[2]:.3f}")
    print(f"Inliers Captured: {cf_mask.sum()} / 100 ({cf_mask[:50].sum()} circle pts + {cf_mask[50:].sum()} line pts)")
    print("=" * 60)

    # Visualizations:
    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.90, wspace=0.40, top=0.92, bottom=0.08)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    xs = np.linspace(-16, 16, 300)
    th = np.linspace(0, 2 * np.pi, 200)

    # Panel 1: Line-First RANSAC (Figure 1 replication)
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c="#94a3b8", s=20, alpha=0.5, label="All points")
    ax.scatter(X[line_mask, 0], X[line_mask, 1], c="#2563eb", s=28, label=f"Line inliers ({line_mask.sum()})")
    ax.scatter(X[circ_global_mask, 0], X[circ_global_mask, 1], c="#16a34a", s=28, label=f"Circle inliers ({circ_global_mask.sum()})")
    # Samples (without edgecolors for line markers 'x' and '+')
    ax.scatter(X[line_sample, 0], X[line_sample, 1], c="#1e3a8a", s=80, marker="x", lw=2.2, label="Best sample for line")
    ax.scatter(X[circ_global_sample, 0], X[circ_global_sample, 1], c="#991b1b", s=80, marker="+", lw=2.2, label="Best sample for circle")

    # Lines and circles
    ax.plot(xs, m_gt * xs + b_gt, color="magenta", ls="--", lw=1.6, label="Ground truth line")
    ax.plot(center_gt[0] + r_gt * np.cos(th), center_gt[1] + r_gt * np.sin(th), color="darkgreen", ls="--", lw=1.6, label="Ground truth circle")
    ax.plot(xs, (line_samp_m[2] - line_samp_m[0] * xs) / line_samp_m[1], color="#60a5fa", ls=":", lw=1.5, label="Line from best sample")
    ax.plot(xs, (line_ref[2] - line_ref[0] * xs) / line_ref[1], color="#1d4ed8", lw=2.0, label="RANSAC line (TLS)")
    ax.plot(circ_samp_m[0] + circ_samp_m[2] * np.cos(th), circ_samp_m[1] + circ_samp_m[2] * np.sin(th), color="#f87171", ls=":", lw=1.5, label="Circle from best sample")
    ax.plot(circ_ref[0] + circ_ref[2] * np.cos(th), circ_ref[1] + circ_ref[2] * np.sin(th), color="#dc2626", lw=2.0, label="RANSAC circle (Nelder-Mead)")

    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.9, edgecolor="#cbd5e1")
    ax.set_title("Figure 2.1: Sequential Line-First RANSAC (Matches Brief Fig. 1)", fontsize=10.0, weight="bold")

    # Panel 2: REAL Circle-First RANSAC
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c="#94a3b8", s=20, alpha=0.5, label="All points")
    # Breakdown of circle-first inliers
    cf_circ_pts = np.where(cf_mask & (np.arange(len(X)) < 50))[0]
    cf_line_pts = np.where(cf_mask & (np.arange(len(X)) >= 50))[0]
    ax.scatter(X[cf_circ_pts, 0], X[cf_circ_pts, 1], c="#16a34a", s=30, label=f"Circle pts captured ({len(cf_circ_pts)})")
    ax.scatter(X[cf_line_pts, 0], X[cf_line_pts, 1], c="#ea580c", s=36, marker="s", label=f"Line pts falsely captured ({len(cf_line_pts)})")
    ax.scatter(X[cf_sample, 0], X[cf_sample, 1], c="#991b1b", s=80, marker="+", lw=2.2, label="Sample triplet")

    ax.plot(xs, m_gt * xs + b_gt, color="magenta", ls="--", lw=1.6, label="Ground truth line")
    ax.plot(center_gt[0] + r_gt * np.cos(th), center_gt[1] + r_gt * np.sin(th), color="darkgreen", ls="--", lw=1.6, label="Ground truth circle")
    ax.plot(cf_ref[0] + cf_ref[2] * np.cos(th), cf_ref[1] + cf_ref[2] * np.sin(th), color="#dc2626", lw=2.0, label=f"Circle-first fit (r={cf_ref[2]:.2f})")

    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.9, edgecolor="#cbd5e1")
    ax.set_title("Figure 2.2: Circle-First RANSAC on Full Data (Intersection Contamination)", fontsize=10.0, weight="bold")

    out_path = output_dir / "line_circle_ransac.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Question 2 plot to: {out_path}")


if __name__ == "__main__":
    out = ROOT / "results" / "question_2"
    run_q2(out)
