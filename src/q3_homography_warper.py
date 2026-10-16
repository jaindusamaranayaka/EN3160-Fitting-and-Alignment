"""Question 3: Planar Flag Superimposition via Homography Warping.

Carries out planar homography warping and alpha blending for a couple of image pairs
of user choice, with plain-terms non-technical rationale:
  - Pair 1: Oxford Wadham College (wadham.jpg) + British Flag (flag.png)
  - Pair 2: Classical Academic Building (building.jpg) + Sri Lankan Flag (flag_sl.png)
Implements Direct Linear Transformation (DLT) via SVD and compares with OpenCV.
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


def compute_homography_dlt(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute 3x3 homography matrix H via Direct Linear Transformation (DLT) using SVD.

    For each point correspondence (x, y) -> (u, v):
      [-x, -y, -1,  0,  0,  0,  u*x,  u*y,  u] * h = 0
      [ 0,  0,  0, -x, -y, -1,  v*x,  v*y,  v] * h = 0
    The 8x9 system A*h = 0 is solved by taking the right singular vector of A
    associated with the smallest singular value (last row of V^T).
    """
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def warp_and_blend(
    base_bgr: np.ndarray,
    flag_bgr: np.ndarray,
    dst_corners: np.ndarray,
    alpha: float = 0.70,
    modulate_luminance: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp flag onto base scene quadrilateral and alpha blend."""
    bh, bw = base_bgr.shape[:2]
    fh, fw = flag_bgr.shape[:2]

    src_corners = np.float32([
        [0, 0],
        [fw - 1, 0],
        [fw - 1, fh - 1],
        [0, fh - 1],
    ])
    dst_corners = np.asarray(dst_corners, dtype=np.float32)

    # Compute homography via DLT
    H = compute_homography_dlt(src_corners, dst_corners)

    warped_flag = cv2.warpPerspective(flag_bgr, H, (bw, bh), flags=cv2.INTER_LINEAR)
    ones = np.ones((fh, fw), dtype=np.uint8) * 255
    mask = cv2.warpPerspective(ones, H, (bw, bh), flags=cv2.INTER_NEAREST) > 0

    blended = base_bgr.copy()
    if modulate_luminance:
        gray_b = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lum = np.clip(np.repeat((gray_b / 0.68)[:, :, np.newaxis], 3, axis=2), 0.45, 1.25)
        flag_shaded = np.clip(warped_flag.astype(np.float32) * lum, 0, 255).astype(np.uint8)
        blended[mask] = cv2.addWeighted(flag_shaded, alpha, base_bgr, 1.0 - alpha, 0.0)[mask]
    else:
        blended[mask] = cv2.addWeighted(
            warped_flag, alpha, base_bgr, 1.0 - alpha, 0.0
        )[mask]

    return blended, warped_flag, H


def run_q3(image_dir: Path, output_dir: Path):
    """Execute Question 3 for a couple of image pairs and save visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pair 1: Wadham College + UK Flag (Replicates Figure 2 in assignment brief)
    wadham_bgr = cv2.imread(str(image_dir / "wadham.jpg"))
    flag_uk = cv2.imread(str(image_dir / "flag.png"))[..., :3]
    wadham_corners = np.float32([
        [147, 208],  # Top-left
        [516, 292],  # Top-right
        [520, 520],  # Bottom-right
        [135, 518],  # Bottom-left
    ])
    blend_1, _, H1 = warp_and_blend(wadham_bgr, flag_uk, wadham_corners, alpha=0.68)

    # Pair 2: Academic Building + Sri Lankan Flag (Cleanly aligned to building facade plane)
    bldg_bgr = cv2.imread(str(image_dir / "building.jpg"))
    flag_sl = cv2.imread(str(image_dir / "flag_sl.png"))[..., :3]
    bldg_corners = np.float32([
        [225, 215],  # Top-left (3rd floor sill line)
        [640, 150],  # Top-right (3rd floor sill line, column 6)
        [640, 395],  # Bottom-right (main concrete beam above ground floor)
        [225, 420],  # Bottom-left (main concrete beam above ground floor)
    ])
    blend_2, _, H2 = warp_and_blend(bldg_bgr, flag_sl, bldg_corners, alpha=0.78, modulate_luminance=True)

    # Verify DLT matches OpenCV's findHomography
    fh1, fw1 = flag_uk.shape[:2]
    src1 = np.float32([[0, 0], [fw1 - 1, 0], [fw1 - 1, fh1 - 1], [0, fh1 - 1]])
    H1_cv, _ = cv2.findHomography(src1, wadham_corners)
    H1_diff = np.max(np.abs(H1 / H1[2, 2] - H1_cv / H1_cv[2, 2]))
    print(f"Max difference between custom DLT (SVD) and OpenCV findHomography: {H1_diff:.2e}")

    # Save individual composite images
    cv2.imwrite(str(output_dir / "wadham_flag_blended.png"), blend_1)
    cv2.imwrite(str(output_dir / "building_flag_sl_blended.png"), blend_2)

    # Generate 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Wadham Quadrangle Selection
    axes[0, 0].imshow(cv2.cvtColor(wadham_bgr, cv2.COLOR_BGR2RGB))
    poly1 = np.vstack([wadham_corners, wadham_corners[0:1]])
    axes[0, 0].plot(poly1[:, 0], poly1[:, 1], "c-", lw=2.2, label="Courtyard Wall Plane")
    axes[0, 0].scatter(wadham_corners[:, 0], wadham_corners[:, 1], c="yellow", s=40, zorder=5)
    axes[0, 0].set_title("Figure 3.1: Pair 1 - Wadham Quad Corners (Figure 2 brief)", fontsize=10.5, weight="bold")
    axes[0, 0].legend(fontsize=9, loc="lower right")
    axes[0, 0].axis("off")

    # Panel 2: Wadham Blended Result
    axes[0, 1].imshow(cv2.cvtColor(blend_1, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Figure 3.2: Pair 1 - British Flag Warped & Blended", fontsize=10.5, weight="bold")
    axes[0, 1].axis("off")

    # Panel 3: Building Facade Selection
    axes[1, 0].imshow(cv2.cvtColor(bldg_bgr, cv2.COLOR_BGR2RGB))
    poly2 = np.vstack([bldg_corners, bldg_corners[0:1]])
    axes[1, 0].plot(poly2[:, 0], poly2[:, 1], "lime", lw=2.2, label="Main Facade Plane")
    axes[1, 0].scatter(bldg_corners[:, 0], bldg_corners[:, 1], c="yellow", s=40, zorder=5)
    axes[1, 0].set_title("Figure 3.3: Pair 2 - Building Facade Corners", fontsize=10.5, weight="bold")
    axes[1, 0].legend(fontsize=9, loc="lower right")
    axes[1, 0].axis("off")

    # Panel 4: Building Blended Result
    axes[1, 1].imshow(cv2.cvtColor(blend_2, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Figure 3.4: Pair 2 - Sri Lankan Flag Warped & Blended", fontsize=10.5, weight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_path = output_dir / "homography_warping_pairs.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Question 3 plot to: {out_path}")

    print("\nNon-Technical Rationale for Image Choices:")
    print("• Pair 1 (Wadham College + British Flag): Replicates Figure 2 from the assignment brief.")
    print("  In plain terms, it displays the Union Jack across the residential quadrangle wall of Oxford's")
    print("  historic 17th-century Wadham College between dorm windows, celebrating British collegiate heritage.")
    print("• Pair 2 (Civic Academic Building + Sri Lankan Flag): Simulates an international cultural festival")
    print("  and visiting delegation reception. In plain terms, it drapes the Sri Lankan national flag across")
    print("  the grand front entrance of a university hall to celebrate international student diversity and partnership.")


if __name__ == "__main__":
    imgs = ROOT / "a2images"
    out = ROOT / "results" / "question_3"
    run_q3(imgs, out)
