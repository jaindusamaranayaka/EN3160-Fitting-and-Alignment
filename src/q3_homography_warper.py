"""Question 3: Planar Flag Superimposition via Homography Warping.

Given an architectural image and a planar surface defined by 4 corner points,
this module computes the homography from flag image coordinates to the planar surface,
warps the flag, and alpha-blends it onto the architectural scene.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path if running directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corner_selector import CALIBRATED_CORNERS, get_corners


def compute_homography_and_warp(
    source_image: np.ndarray,
    destination_corners: np.ndarray,
    target_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute homography from source image rectangle to destination corners and warp.

    Args:
        source_image: BGR or BGRA source image
        destination_corners: (4, 2) array of destination points [TL, TR, BR, BL]
        target_shape: (height, width) of the target destination scene

    Returns:
        Tuple of (warped_image, binary_mask, homography_matrix)
    """
    src_h, src_w = source_image.shape[:2]
    # Source corners corresponding to [top-left, top-right, bottom-right, bottom-left]
    src_corners = np.float32([
        [0, 0],
        [src_w - 1, 0],
        [src_w - 1, src_h - 1],
        [0, src_h - 1],
    ])

    dst_corners = np.asarray(destination_corners, dtype=np.float32)

    homography, status = cv2.findHomography(src_corners, dst_corners, 0)
    if homography is None:
        raise ValueError("Could not compute homography from the provided corners.")

    target_h, target_w = target_shape
    warped = cv2.warpPerspective(
        source_image,
        homography,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Compute binary mask of warped valid region
    ones_mask = np.ones((src_h, src_w), dtype=np.float32)
    warped_mask = cv2.warpPerspective(
        ones_mask,
        homography,
        (target_w, target_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask = (warped_mask > 0.5).astype(np.float32)

    return warped, mask, homography


def blend_images(
    base_image: np.ndarray,
    warped_image: np.ndarray,
    warped_mask: np.ndarray,
    alpha_blend: float = 0.70,
) -> np.ndarray:
    """Alpha-blend the warped foreground onto the base image.

    Args:
        base_image: BGR background image (uint8)
        warped_image: BGR or BGRA warped foreground (uint8)
        warped_mask: Binary float mask (0.0 or 1.0) of warped region
        alpha_blend: Opacity of the foreground within the mask (0.0 to 1.0)

    Returns:
        BGR blended composite image (uint8)
    """
    # If warped has 4 channels, incorporate its intrinsic alpha channel
    if warped_image.shape[2] == 4:
        intrinsic_alpha = warped_image[:, :, 3].astype(np.float32) / 255.0
        effective_alpha = intrinsic_alpha * warped_mask * alpha_blend
        fg = warped_image[:, :, :3].astype(np.float32)
    else:
        effective_alpha = warped_mask * alpha_blend
        fg = warped_image.astype(np.float32)

    effective_alpha = effective_alpha[:, :, np.newaxis]
    bg = base_image.astype(np.float32)

    composite = fg * effective_alpha + bg * (1.0 - effective_alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


def run_flag_warping(
    architecture_path: Path,
    flag_path: Path,
    output_dir: Path,
    interactive: bool = False,
):
    """Run full Question 3 pipeline for multiple planar placements."""
    output_dir.mkdir(parents=True, exist_ok=True)

    arch_bgr = cv2.imread(str(architecture_path))
    flag_bgr = cv2.imread(str(flag_path), cv2.IMREAD_UNCHANGED)

    if arch_bgr is None:
        raise FileNotFoundError(f"Could not load architecture image: {architecture_path}")
    if flag_bgr is None:
        raise FileNotFoundError(f"Could not load flag image: {flag_path}")

    h, w = arch_bgr.shape[:2]

    placements = [
        ("wadham_left_wall", 1, "Placement 1 (Left Courtyard Facade - Figure 2 Replication)", 0.65),
        ("wadham_lawn", 2, "Placement 2 (Courtyard Lawn - Ceremonial Display)", 0.75),
        ("wadham_right_wall", 3, "Placement 3 (Right Chapel Facade)", 0.65),
    ]

    results = []

    # Overlay figure showing all selected planar surfaces
    fig_overlay, ax_overlay = plt.subplots(figsize=(12, 8))
    ax_overlay.imshow(cv2.cvtColor(arch_bgr, cv2.COLOR_BGR2RGB))
    colors = ["cyan", "yellow", "lime"]

    for idx, (preset_key, p_num, label, alpha_val) in enumerate(placements):
        print(f"\n--- Processing {label} ---")
        corners = get_corners(
            arch_bgr,
            preset_key=preset_key,
            interactive=interactive,
            window_title=f"Corner Selection: {label}",
        )

        # Plot polygon on overlay
        poly = np.vstack([corners, corners[0:1]])
        ax_overlay.plot(poly[:, 0], poly[:, 1], color=colors[idx], linewidth=2.5, label=label)
        for i_pt, pt in enumerate(corners):
            ax_overlay.plot(pt[0], pt[1], "o", color=colors[idx], markersize=8)
            ax_overlay.text(pt[0] + 8, pt[1] - 8, f"P{p_num}_{i_pt+1}", color="white",
                            fontsize=9, weight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[idx], alpha=0.7))

        warped, mask, H = compute_homography_and_warp(flag_bgr, corners, (h, w))
        blended = blend_images(arch_bgr, warped, mask, alpha_blend=alpha_val)

        out_path = output_dir / f"wadham_flag_{p_num}.png"
        cv2.imwrite(str(out_path), blended)
        print(f"Saved: {out_path}")
        results.append((p_num, label, corners, H, blended))

    ax_overlay.set_title("Wadham College: Calibrated Planar Placement Surfaces (Homography Boundaries)", fontsize=13)
    ax_overlay.legend(loc="lower right", fontsize=10)
    ax_overlay.axis("off")
    plt.tight_layout()
    overlay_path = output_dir / "wadham_corners_overlay.png"
    fig_overlay.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close(fig_overlay)
    print(f"Saved corners overlay to: {overlay_path}")

    return results


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    arch = proj_root / "a2images" / "wadham.jpg"
    flag = proj_root / "a2images" / "flag.png"
    out = proj_root / "results" / "question_3"
    run_flag_warping(arch, flag, out, interactive=False)
