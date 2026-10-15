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
) -> Tuple[np.ndarray, List[Tuple[float, int, int, float, float]], float]:
    """Detect sunflower centers using scale-normalized Laplacian of Gaussian.

    Sunflower flower heads have dark central disks (brown/black) surrounded by
    bright yellow petals. For a dark blob on a bright background, the Laplacian
    second derivatives are positive (+Laplacian), reaching peak scale-normalized
    response at sigma = r / sqrt(2), i.e., r = sqrt(2) * sigma.

    Args:
        image_bgr: Input BGR image (loaded with cv2.IMREAD_REDUCED_COLOR_4)
        min_sigma: Smallest scale standard deviation (pixels)
        max_sigma: Largest scale standard deviation (pixels)
        num_scales: Number of scales geometrically sampled
        max_detections: Maximum circles to retain

    Returns:
        sigmas: 1D array of sigma scales
        detections: List of (score, x, y, sigma, radius)
        response_floor: Threshold used for candidate selection
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 1. Sunflower field region gating via yellow petal envelope
    petal_mask = cv2.inRange(hsv, np.array([14, 70, 70]), np.array([36, 255, 255]))
    disk_roi = cv2.dilate(petal_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)

    # 2. Scale-space positive normalized Laplacian (dark-blob detector)
    sigmas = np.geomspace(min_sigma, max_sigma, num_scales)
    responses = []
    for scale in sigmas:
        blurred = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=float(scale), sigmaY=float(scale))
        # Positive normalized Laplacian peaks at centers of dark blobs
        response = (float(scale) ** 2) * cv2.Laplacian(blurred, cv2.CV_32F)
        responses.append(response)
    scale_space = np.stack(responses, axis=0)

    # 3. Find 3D local extrema (scale, y, x)
    local_max = scale_space == maximum_filter(scale_space, size=(3, 5, 5), mode="nearest")
    response_floor = 0.035

    candidates = []
    horizon_cutoff = int(0.51 * h)  # Exclude sky and horizon tree line
    for k in range(len(sigmas)):
        s = float(sigmas[k])
        r = float(np.sqrt(2) * s)
        pts = np.argwhere(local_max[k] & (scale_space[k] >= response_floor))
        for y, x in pts:
            if y < horizon_cutoff:
                continue
            if disk_roi[y, x] == 0:
                continue
            if gray_float[y, x] > 0.48:  # Center disk must be dark
                continue
            score = float(scale_space[k, y, x])
            candidates.append((score, int(x), int(y), s, r))

    candidates.sort(reverse=True)

    # 4. Spatial non-maximum suppression (ensures single circle per sunflower)
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
        if len(detections) >= max_detections:
            break

    return sigmas, detections, response_floor


def run_q1(image_path: Path, output_dir: Path):
    """Run Question 1 and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_4)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sigmas, detections, floor = detect_sunflowers(image_bgr)

    print(f"Scale range: sigma = {sigmas.min():.2f} to {sigmas.max():.2f} px ({len(sigmas)} scales)")
    print(f"Response threshold: {floor:.4f}; Total detections: {len(detections)}")
    print("\nLargest detections (x, y, sigma, radius, response):")
    largest = sorted(detections, key=lambda item: item[4], reverse=True)[:10]
    for idx, (score, x, y, scale, radius) in enumerate(largest, 1):
        print(f"{idx:2d}. Center: ({x:4d}, {y:4d}), sigma={scale:5.2f} px, radius={radius:5.2f} px, response={score:.4f}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_rgb)
    for score, x, y, scale, radius in detections:
        ax.add_patch(plt.Circle((x, y), radius, color="lime", fill=False, linewidth=1.5))
    ax.set_title(f"Sunflower centers detected by scale-space LoG ({len(detections)} detections)")
    ax.axis("off")
    plt.tight_layout()

    out_path = output_dir / "sunflower_log_detections.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Question 1 detection plot to: {out_path}")


if __name__ == "__main__":
    img = ROOT / "a2images" / "the_berry_farms_sunflower_field.jpeg"
    out = ROOT / "results" / "question_1"
    run_q1(img, out)
