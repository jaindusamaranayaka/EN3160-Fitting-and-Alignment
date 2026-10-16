"""Question 1: Sunflower Detection via Scale-Space Laplacian of Gaussian (LoG).

Detects sunflower centers (dark disks on bright yellow flower heads) using
scale-space local maxima of scale-normalized Laplacian of Gaussian:
    LoG_norm(x, y; sigma) = +sigma^2 * Laplacian(G_sigma * I)
Applies sunflower ROI gating and spatial non-maximum suppression (NMS) to fit
exactly one circle (r = sqrt(2) * sigma) to each flower head center, reports
scale range and largest detections, and saves visualization to results/question_1/.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def detect_sunflowers(
    image_bgr: np.ndarray,
    min_sigma: float = 1.5,
    max_sigma: float = 12.5,
    num_scales: int = 24,
    max_detections: int = 50,
) -> Tuple[np.ndarray, List[Tuple[float, int, int, float, float]], float, int]:
    """Detect sunflower centers using scale-normalized Laplacian of Gaussian.

    Sunflower flower heads have dark central disks (brown/black) surrounded by
    bright yellow petals. For a dark blob on a bright background, the Laplacian
    second derivatives are positive (+Laplacian), reaching peak scale-normalized
    response at sigma = r / sqrt(2), i.e., r = sqrt(2) * sigma.

    All coordinates and radii are in the 1/4-reduced image coordinate space
    (cv2.IMREAD_REDUCED_COLOR_4).

    Four explicit filters are applied:
      1. Row cutoff: y >= int(0.51 * H) isolates ground field below horizon.
      2. HSV petal envelope: yellow petal mask (H in [14, 36], S >= 70, V >= 70)
         dilated with 17x17 ellipse restricts candidates to flower heads.
      3. Grayscale dark intensity: gray_float[y, x] <= 0.48 ensures candidate is
         the dark central floret disk, not bright yellow petals or sky.
      4. LoG response threshold: scale-space local maximum response >= 0.035.
      Boundary filter: 6 <= x <= W - 6 and 6 <= y <= H - 6 eliminates border artefacts.

    Args:
        image_bgr: Input BGR image (loaded with cv2.IMREAD_REDUCED_COLOR_4)
        min_sigma: Smallest scale standard deviation (pixels)
        max_sigma: Largest scale standard deviation (pixels)
        num_scales: Number of scales geometrically sampled
        max_detections: Number of top detections to return (0 for all)

    Returns:
        sigmas: 1D array of sigma scales
        detections: List of (score, x, y, sigma, radius)
        response_floor: Threshold used for candidate selection
        total_candidates: Total scale-space candidates before NMS/capping
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Filter 2: Sunflower field region gating via yellow petal envelope
    petal_mask = cv2.inRange(hsv, np.array([14, 70, 70]), np.array([36, 255, 255]))
    disk_roi = cv2.dilate(
        petal_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )

    # Scale-space positive normalized Laplacian (dark-blob detector)
    sigmas = np.geomspace(min_sigma, max_sigma, num_scales)
    responses = []
    for scale in sigmas:
        blurred = cv2.GaussianBlur(
            gray_float, (0, 0), sigmaX=float(scale), sigmaY=float(scale)
        )
        response = (float(scale) ** 2) * cv2.Laplacian(blurred, cv2.CV_32F)
        responses.append(response)
    scale_space = np.stack(responses, axis=0)

    # Find 3D local extrema (scale, y, x)
    local_max = scale_space == maximum_filter(
        scale_space, size=(3, 5, 5), mode="nearest"
    )
    response_floor = 0.035

    candidates = []
    horizon_cutoff = int(0.51 * h)  # Filter 1: Row cutoff below horizon
    for k in range(len(sigmas)):
        s = float(sigmas[k])
        r = float(np.sqrt(2) * s)
        pts = np.argwhere(local_max[k] & (scale_space[k] >= response_floor))
        for y, x in pts:
            # Boundary margin: reject border artefacts
            if x < 6 or x > w - 6 or y < 6 or y > h - 6:
                continue
            if y < horizon_cutoff:  # Filter 1
                continue
            if disk_roi[y, x] == 0:  # Filter 2
                continue
            if gray_float[y, x] > 0.48:  # Filter 3: Dark center disk
                continue
            score = float(scale_space[k, y, x])
            candidates.append((score, int(x), int(y), s, r))

    total_candidates = len(candidates)
    candidates.sort(reverse=True)

    # Spatial non-maximum suppression (ensures single circle per sunflower)
    detections: List[Tuple[float, int, int, float, float]] = []
    for score, x, y, scale, radius in candidates:
        overlap = False
        for _, dx, dy, _, dr in detections:
            dist = float(np.hypot(x - dx, y - dy))
            if dist < 0.85 * (radius + dr) or dist < 1.10 * max(radius, dr):
                overlap = True
                break
        if not overlap:
            detections.append((score, x, y, scale, radius))
        if max_detections > 0 and len(detections) >= max_detections:
            break

    return sigmas, detections, response_floor, total_candidates


def run_q1(image_path: Path, output_dir: Path):
    """Run Question 1 and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_4)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sigmas, detections, floor, total_cand = detect_sunflowers(
        image_bgr, max_detections=50
    )

    print(
        f"Coordinates are in 1/4-reduced image space ({image_bgr.shape[1]}x{image_bgr.shape[0]})."
    )
    print(
        f"Scale range: sigma = {sigmas.min():.2f} to {sigmas.max():.2f} px ({len(sigmas)} scales)"
    )
    print(
        f"Characteristic radius range: r = {np.sqrt(2)*sigmas.min():.2f} to {np.sqrt(2)*sigmas.max():.2f} px"
    )
    print(f"Total candidate extrema: {total_cand}; Top detections displayed: {len(detections)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Detection overlay
    axes[0].imshow(image_rgb)
    for score, x, y, scale, radius in detections:
        axes[0].add_patch(
            plt.Circle((x, y), radius, color="lime", fill=False, linewidth=1.4)
        )
    axes[0].set_title(
        f"Figure 1.1: Top {len(detections)} Sunflower Disks by LoG Response (1/4-scale)",
        fontsize=11,
        weight="bold",
    )
    axes[0].axis("off")

    # Radius vs row y depth trend
    ys = np.array([d[2] for d in detections])
    rs = np.array([d[4] for d in detections])
    axes[1].scatter(ys, rs, c="#1e3a8a", s=30, alpha=0.8, edgecolors="none")
    slope, intercept = np.polyfit(ys, rs, 1)
    y_line = np.linspace(ys.min(), ys.max(), 50)
    axes[1].plot(
        y_line,
        slope * y_line + intercept,
        "r--",
        lw=1.8,
        label=f"Linear fit: r = {slope:.3f}y {intercept:+.2f} (R = {np.corrcoef(ys, rs)[0,1]:.2f})",
    )
    axes[1].set_xlabel("Image Row y (px, increasing towards camera)", fontsize=10)
    axes[1].set_ylabel("Fitted Disk Radius r = sqrt(2)*sigma (px)", fontsize=10)
    axes[1].set_title(
        "Figure 1.2: Perspective Depth Scaling (Radius vs Row y)",
        fontsize=11,
        weight="bold",
    )
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out_path = output_dir / "sunflower_log_detections.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Question 1 plot to: {out_path}")


if __name__ == "__main__":
    img = ROOT / "a2images" / "the_berry_farms_sunflower_field.jpeg"
    out = ROOT / "results" / "question_1"
    run_q1(img, out)
