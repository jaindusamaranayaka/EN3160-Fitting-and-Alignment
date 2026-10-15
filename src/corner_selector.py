"""Interactive Corner Selector Utility for EN3160 Assignment 2.

Provides an interactive GUI for selecting 4 planar corners on an image
using OpenCV Win32 UI (or Matplotlib fallback), with live visual feedback,
crosshairs, undo, reset, and confirmation.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


# Pre-calibrated default corners for wadham.jpg
CALIBRATED_CORNERS = {
    "wadham_left_wall": np.float32([
        [147, 208],  # 1: Top-Left (near roofline / left buttress)
        [516, 292],  # 2: Top-Right (courtyard corner crease)
        [520, 520],  # 3: Bottom-Right (courtyard corner base)
        [135, 518],  # 4: Bottom-Left (ground ledge)
    ]),
    "wadham_lawn": np.float32([
        [150, 600],  # 1: Top-Left (rear lawn boundary)
        [874, 600],  # 2: Top-Right (rear lawn boundary)
        [980, 750],  # 3: Bottom-Right (foreground lawn boundary)
        [44, 750],   # 4: Bottom-Left (foreground lawn boundary)
    ]),
    "wadham_right_wall": np.float32([
        [520, 295],  # 1: Top-Left (courtyard corner crease)
        [960, 235],  # 2: Top-Right (roof parapet)
        [960, 680],  # 3: Bottom-Right (ground ledge)
        [520, 520],  # 4: Bottom-Left (courtyard corner base)
    ]),
}


def select_corners_opencv(
    image: np.ndarray,
    window_title: str = "Select 4 Corners (TL -> TR -> BR -> BL)",
) -> Optional[np.ndarray]:
    """Open an interactive OpenCV window to select 4 corner points.

    Controls:
        - Left Click: Add a point (in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        - 'u' or 'z': Undo last point
        - 'r': Reset all points
        - 'c' or Enter: Confirm selection (when 4 points are chosen)
        - 'q' or Esc: Cancel / exit without saving

    Args:
        image: BGR or RGB image (uint8)
        window_title: Title of OpenCV GUI window

    Returns:
        np.ndarray of shape (4, 2) with float32 coordinates, or None if cancelled.
    """
    points: List[Tuple[int, int]] = []
    mouse_pos: List[int] = [-1, -1]
    display_img = image.copy()
    h, w = image.shape[:2]

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, mouse_pos
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))

    window_name = window_title
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    corner_labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]

    while True:
        canvas = display_img.copy()

        # Draw crosshairs at current cursor position
        mx, my = mouse_pos
        if 0 <= mx < w and 0 <= my < h:
            cv2.line(canvas, (mx, 0), (mx, h), (180, 180, 180), 1, cv2.LINE_AA)
            cv2.line(canvas, (0, my), (w, my), (180, 180, 180), 1, cv2.LINE_AA)
            coord_text = f"({mx}, {my})"
            cv2.putText(canvas, coord_text, (mx + 10, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw connecting lines and points
        for i, pt in enumerate(points):
            cv2.circle(canvas, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{i+1}", (pt[0] + 10, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            if i > 0:
                cv2.line(canvas, points[i - 1], pt, (0, 255, 0), 2, cv2.LINE_AA)
        if len(points) == 4:
            cv2.line(canvas, points[3], points[0], (0, 255, 0), 2, cv2.LINE_AA)

        # Status overlay banner
        banner_color = (30, 30, 30)
        cv2.rectangle(canvas, (0, 0), (w, 36), banner_color, -1)
        if len(points) < 4:
            prompt = f"Click corner {len(points)+1}/4: {corner_labels[len(points)]} | Keys: [u]ndo, [r]eset, [q]uit"
        else:
            prompt = "All 4 corners selected! Press [c] or [Enter] to confirm, [u]ndo, [r]eset"
        cv2.putText(canvas, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('c'), 13):  # 'c' or Enter
            if len(points) == 4:
                break
        elif key in (ord('u'), ord('z')):  # Undo
            if points:
                points.pop()
        elif key == ord('r'):  # Reset
            points.clear()
        elif key in (ord('q'), 27):  # 'q' or Esc
            points = []
            break

    cv2.destroyWindow(window_name)
    if len(points) == 4:
        return np.asarray(points, dtype=np.float32)
    return None


def get_corners(
    image: np.ndarray,
    preset_key: str = "wadham_left_wall",
    interactive: bool = False,
    window_title: str = "Corner Selection",
) -> np.ndarray:
    """Get 4 corners via live interactive selection or calibrated preset fallback.

    Args:
        image: Architecture image (BGR)
        preset_key: Key in CALIBRATED_CORNERS to use if interactive is False or cancelled
        interactive: If True, attempts to open an interactive OpenCV window
        window_title: Title for the interactive window

    Returns:
        np.ndarray of shape (4, 2) with corner coordinates
    """
    if interactive:
        try:
            selected = select_corners_opencv(image, window_title=window_title)
            if selected is not None and len(selected) == 4:
                print(f"Successfully selected 4 corners interactively:\n{selected}")
                return selected
            print("Interactive selection cancelled or incomplete. Falling back to calibrated preset.")
        except Exception as e:
            print(f"Interactive window could not be displayed ({e}). Using calibrated preset.")

    preset = CALIBRATED_CORNERS.get(preset_key, CALIBRATED_CORNERS["wadham_left_wall"])
    print(f"Using calibrated preset for '{preset_key}':\n{preset}")
    return preset.copy()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Interactive Corner Selector")
    parser.add_argument("--image", type=str, default="a2images/wadham.jpg", help="Path to input image")
    parser.add_argument("--preset", type=str, default="wadham_left_wall", help="Preset fallback key")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    bgr = cv2.imread(str(img_path))
    corners = select_corners_opencv(bgr)
    if corners is not None:
        print("\nSelected corner coordinates:")
        print(repr(corners))
    else:
        print("\nSelection was cancelled.")
