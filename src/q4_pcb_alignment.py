"""Question 4: PCB Alignment and Defect/Difference Inspection via SIFT and RANSAC Homography.

Aligns candidate PCB images against template AECC1.jpg using:
(a) SIFT feature extraction
(b) BFMatcher + Lowe's ratio test + RANSAC homography estimation
(c) Perspective warping and absolute grayscale difference computation
(d) Saves aligned, matches, and difference images to results/question_4/
(e) Explains high difference regions (USB port, capacitors, reset buttons).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def align_and_compare(
    template_bgr: np.ndarray,
    pcb_bgr: np.ndarray,
    ratio_thresh: float = 0.75,
    ransac_reproj_thresh: float = 5.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    """Align PCB to template using SIFT and compute absolute difference.

    Returns:
        (matches_image, aligned_image, difference_image, num_good_matches, num_inliers)
    """
    sift = cv2.SIFT_create()
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    pcb_gray = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2GRAY)

    kp_tpl, des_tpl = sift.detectAndCompute(template_gray, None)
    kp_pcb, des_pcb = sift.detectAndCompute(pcb_gray, None)

    if des_tpl is None or des_pcb is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des_tpl, des_pcb, k=2)

    good_matches = [
        m for pair in knn_matches if len(pair) == 2
        for m, n in [pair] if m.distance < ratio_thresh * n.distance
    ]

    if len(good_matches) < 4:
        return None

    pts_tpl = np.float32([kp_tpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_pcb = np.float32([kp_pcb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography mapping PCB -> Template
    H, inlier_mask = cv2.findHomography(pts_pcb, pts_tpl, cv2.RANSAC, ransac_reproj_thresh)
    if H is None or inlier_mask is None:
        return None

    # Draw inlier matches
    matches_img = cv2.drawMatches(
        template_bgr, kp_tpl, pcb_bgr, kp_pcb, good_matches, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        matchesMask=inlier_mask.ravel().tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    h_tpl, w_tpl = template_gray.shape
    aligned = cv2.warpPerspective(pcb_bgr, H, (w_tpl, h_tpl))
    diff = cv2.absdiff(template_gray, cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))

    return matches_img, aligned, diff, len(good_matches), int(inlier_mask.sum())


def run_q4(pcb_dir: Path, output_dir: Path):
    """Run Question 4 on all supplied PCB boards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = pcb_dir / "AECC1.jpg"
    template_bgr = cv2.imread(str(template_path))
    if template_bgr is None:
        raise FileNotFoundError(f"Could not load template PCB: {template_path}")

    pcb_names = ["AAAA1", "ADAD1", "AECC2", "AECC3", "AECC4"]
    for name in pcb_names:
        pcb_path = pcb_dir / f"{name}.jpg"
        pcb_bgr = cv2.imread(str(pcb_path))
        if pcb_bgr is None:
            print(f"Skipping {name}: file not found.")
            continue

        res = align_and_compare(template_bgr, pcb_bgr)
        if res is None:
            print(f"Failed to align {name}")
            continue

        matches_img, aligned, diff, n_good, n_inliers = res

        cv2.imwrite(str(output_dir / f"{name}_matches.png"), matches_img)
        cv2.imwrite(str(output_dir / f"{name}_aligned.png"), aligned)
        cv2.imwrite(str(output_dir / f"{name}_difference.png"), diff)

        print(f"{name}: {n_good} ratio matches, {n_inliers} RANSAC inliers -> saved to {output_dir}")

    print("\nQuestion 4(e) Analysis:")
    print("Prominent components such as the USB Type B connector, electrolytic capacitors,")
    print("and tactile reset buttons produce strong difference residuals because their physical")
    print("positions, heights, component models, or presence vary across distinct board variants.")
    print("Planar homography corrects perspective for coplanar circuit board surfaces, but cannot")
    print("compensate for 3D component parallax, distinct component types, or absent parts.")


if __name__ == "__main__":
    pcbs = ROOT / "a2images" / "pcbs"
    out = ROOT / "results" / "question_4"
    run_q4(pcbs, out)
