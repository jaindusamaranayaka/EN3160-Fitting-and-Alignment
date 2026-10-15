"""Question 1: Sunflower Detection via Scale-Space Laplacian of Gaussian (LoG).

Detects sunflower centers using scale-space extrema of scale-normalized LoG,
applies non-maximum suppression, reports scale range and largest detections,
and saves visualization to results/question_1/.
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
    max_sigma: float = 12.0,
    num_scales: int = 18,
    max_detections: int = 100,
) -> Tuple[np.ndarray, List[Tuple[float, int, int, float, float]], float]:
    """Detect sunflower centers using scale-normalized Laplacian of Gaussian.

    Args:
        image_bgr: Input BGR image
        min_sigma: Smallest scale standard deviation (pixels)
        max_sigma: Largest scale standard deviation (pixels)
        num_scales: Number of scales geometrically sampled
        max_detections: Maximum circles to retain

    Returns:
        sigmas: 1D array of sigma scales
        detections: List of (score, x, y, sigma, radius)
        response_floor: Threshold used for candidate selection
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0

    sigmas = np.geomspace(min_sigma, max_sigma, num_scales)
    responses = []
    for scale in sigmas:
        blurred = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=float(scale), sigmaY=float(scale))
        # Negative scale-normalized Laplacian for blob response
        response = -(float(scale) ** 2) * cv2.Laplacian(blurred, cv2.CV_32F)
        responses.append(response)
    scale_space = np.stack(responses, axis=0)

    # Find 3D local extrema (scale, y, x)
    local_max = scale_space == maximum_filter(scale_space, size=(3, 5, 5), mode="nearest")
    positive = scale_space[scale_space > 0]
    response_floor = max(float(np.quantile(positive, 0.997)), 0.025) if positive.size else 0.025

    candidate_indices = np.argwhere(local_max & (scale_space >= response_floor))
    candidates = [
        (float(scale_space[k, y, x]), int(x), int(y), float(sigmas[k]), float(np.sqrt(2) * sigmas[k]))
        for k, y, x in candidate_indices
        if y >= int(0.50 * gray.shape[0])
    ]
    candidates.sort(reverse=True)

    # Spatial non-maximum suppression
    detections: List[Tuple[float, int, int, float, float]] = []
    for score, x, y, scale, radius in candidates:
        if all((x - dx) ** 2 + (y - dy) ** 2 >= (1.4 * max(radius, dr)) ** 2
               for _, dx, dy, _, dr in detections):
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
    for score, x, y, scale, radius in largest:
        print(f"({x:4d}, {y:4d}), sigma={scale:5.2f}, r={radius:5.2f}, response={score:.4f}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_rgb)
    for score, x, y, scale, radius in detections:
        ax.add_patch(plt.Circle((x, y), radius, color="lime", fill=False, linewidth=1.4))
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
