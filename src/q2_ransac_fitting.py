"""Question 2: Robust Line and Circle Fitting using RANSAC.

Generates a noisy point set conforming to a line and a circle, performs:
(a) 2-point RANSAC line estimation with ||[a,b]|| = 1 constraint and TLS refitting.
(b) Consensus subtraction and 3-point RANSAC circle estimation with radial error and non-linear refitting.
(c) Generates and saves unified visualization matching Figure 1 of the assignment brief.
(d) Discussion on the effect of fitting order.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_noisy_data(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]:
    """Generate noisy point set conforming to a ground-truth line and circle."""
    rng = np.random.default_rng(seed)
    N = 100
    half_n = N // 2

    # Ground truth circle
    circle_center_gt = np.array([2.0, 3.0])
    circle_radius_gt = 10.0
    s_circ = circle_radius_gt / 16.0
    t = rng.uniform(0, 2 * np.pi, half_n)
    radial_noise = s_circ * rng.standard_normal(half_n)
    X_circ = np.column_stack((
        circle_center_gt[0] + (circle_radius_gt + radial_noise) * np.cos(t),
        circle_center_gt[1] + (circle_radius_gt + radial_noise) * np.sin(t),
    ))

    # Ground truth line: y = m*x + b => -1*x + 2
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
    d = float(normal @ p1)
    return float(normal[0]), float(normal[1]), d


def ransac_line(
    points: np.ndarray,
    threshold: float = 1.0,
    min_inliers: int = 35,
    iterations: int = 2000,
    seed: int = 0,
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray]:
    """Estimate best line model using 2-point RANSAC and Total Least Squares refitting."""
    rng = np.random.default_rng(seed)
    best_model, best_mask, best_sample = None, None, None

    for _ in range(iterations):
        sample = rng.choice(len(points), size=2, replace=False)
        model = line_from_two_points(points[sample[0]], points[sample[1]])
        if model is None:
            continue
        a, b, d = model
        errors = np.abs(points @ np.array([a, b]) - d)
        mask = errors < threshold
        if best_mask is None or mask.sum() > best_mask.sum():
            best_model, best_mask, best_sample = model, mask, sample

    if best_mask is None or best_mask.sum() < min_inliers:
        raise RuntimeError("RANSAC could not find line with sufficient inliers.")

    # Total Least Squares refit on consensus set using SVD
    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = float(normal @ centroid)
    refined_model = (float(normal[0]), float(normal[1]), d)

    return refined_model, best_mask, best_sample


def circle_from_three_points(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Compute circle (cx, cy, radius) passing through three non-collinear points."""
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


def ransac_circle(
    points: np.ndarray,
    threshold: float = 0.8,
    min_inliers: int = 30,
    iterations: int = 4000,
    seed: int = 1,
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Estimate circle using 3-point RANSAC and non-linear least squares refitting."""
    rng = np.random.default_rng(seed)
    best_model, best_mask, best_sample = None, None, None

    for _ in range(iterations):
        sample = rng.choice(len(points), size=3, replace=False)
        model = circle_from_three_points(points[sample])
        if model is None:
            continue
        cx, cy, radius = model
        errors = np.abs(np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)
        mask = errors < threshold
        if best_mask is None or mask.sum() > best_mask.sum():
            best_model, best_mask, best_sample = model, mask, sample

    if best_mask is None or best_mask.sum() < min_inliers:
        raise RuntimeError("RANSAC could not find circle with sufficient inliers.")

    # Refine using scipy.optimize.least_squares on consensus set
    inliers = points[best_mask]
    fit = least_squares(
        lambda params: np.linalg.norm(inliers - params[:2], axis=1) - params[2],
        x0=np.asarray(best_model),
        bounds=([-np.inf, -np.inf, 1e-8], [np.inf, np.inf, np.inf]),
    )
    refined_model = (float(fit.x[0]), float(fit.x[1]), float(fit.x[2]))
    return refined_model, best_mask, best_sample, best_model


def draw_line(ax, model: Tuple[float, float, float], x_vals: np.ndarray, **kwargs):
    """Draw line given normal parameterization a*x + b*y = d."""
    a, b, d = model
    if abs(b) > 1e-8:
        ax.plot(x_vals, (d - a * x_vals) / b, **kwargs)
    else:
        ax.axvline(d / a, **kwargs)


def run_q2(output_dir: Path):
    """Execute Question 2 pipeline and save plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X, center_gt, r_gt, m_gt, b_gt = generate_noisy_data(seed=0)

    # 1. Fit line with RANSAC
    line_model, line_inliers, line_sample = ransac_line(X, threshold=1.0, min_inliers=35)
    la, lb, ld = line_model
    print(f"Refined Line: {la:.4f} x + {lb:.4f} y = {ld:.4f}")
    print(f"Line Inliers: {line_inliers.sum()}/{len(X)} points")

    # 2. Subtract line consensus and fit circle
    remnant = X[~line_inliers]
    circle_model, circ_local_mask, circ_sample, circ_sample_model = ransac_circle(
        remnant, threshold=0.8, min_inliers=30
    )
    circ_inliers = np.zeros(len(X), dtype=bool)
    circ_inliers[np.flatnonzero(~line_inliers)[circ_local_mask]] = True
    cx, cy, cr = circle_model
    print(f"Refined Circle: Center=({cx:.3f}, {cy:.3f}), Radius={cr:.3f}")
    print(f"Circle Inliers: {circ_inliers.sum()}/{len(remnant)} line-subtracted points")

    # 3. Unified Figure matching Figure 1 of assignment brief
    line_sample_model = line_from_two_points(X[line_sample[0]], X[line_sample[1]])
    fig, ax = plt.subplots(figsize=(9, 9))

    # All points
    ax.scatter(X[:, 0], X[:, 1], s=18, color="0.75", label="All points")
    # Inliers
    ax.scatter(X[line_inliers, 0], X[line_inliers, 1], s=24, color="tab:blue", label="Line inliers")
    ax.scatter(X[circ_inliers, 0], X[circ_inliers, 1], s=24, color="tab:orange", label="Circle inliers")

    # Best sample points
    ax.scatter(X[line_sample, 0], X[line_sample, 1], s=95, marker="x", color="navy", label="Best sample for line")
    circ_sample_pts = remnant[circ_sample]
    ax.scatter(circ_sample_pts[:, 0], circ_sample_pts[:, 1], s=90, marker="+", color="darkred", label="Best sample for circle")

    # Lines & Circles
    x_span = np.linspace(-15, 15, 300)
    draw_line(ax, line_sample_model, x_span, color="navy", linestyle=":", linewidth=1.5, label="Line from best sample")
    draw_line(ax, line_model, x_span, color="blue", linewidth=2.0, label="RANSAC line")

    scx, scy, sr = circ_sample_model
    if sr < 100:
        ax.add_patch(plt.Circle((scx, scy), sr, fill=False, color="darkred", linestyle=":", linewidth=1.5, label="Circle from best sample"))

    ax.add_patch(plt.Circle((cx, cy), cr, fill=False, color="red", linewidth=2.0, label="RANSAC circle"))
    ax.add_patch(plt.Circle(tuple(center_gt), r_gt, fill=False, color="green", linestyle="--", linewidth=1.5, label="Ground truth circle"))
    ax.plot(center_gt[0], center_gt[1], "+", color="green", markersize=10)
    ax.plot(x_span, m_gt * x_span + b_gt, color="magenta", linestyle="--", linewidth=1.5, label="Ground truth line")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("RANSAC Line and Circle Fitting", fontsize=13)
    ax.grid(True)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()

    out_path = output_dir / "line_circle_ransac.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Question 2 plot to: {out_path}")


if __name__ == "__main__":
    out = ROOT / "results" / "question_2"
    run_q2(out)
