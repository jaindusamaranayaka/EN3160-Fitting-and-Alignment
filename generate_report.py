"""Script to generate publication-quality 4-page PDF report for EN3160 Assignment 2.

Author: Jaindu Samaranayaka
Index: 230563H
Output: report/230563H_a02.pdf (Strictly 4 pages)
"""

from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = REPORT_DIR / "230563H_a02.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.8,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "figure.dpi": 200,
})

PAGE_WIDTH = 8.27   # A4 width in inches
PAGE_HEIGHT = 11.69 # A4 height in inches


def add_header(fig, title: str, subtitle: str = ""):
    """Add a clean, professional header to each page."""
    header_ax = fig.add_axes([0.06, 0.925, 0.88, 0.055])
    header_ax.axis("off")
    header_ax.text(0.0, 0.65, title, fontsize=12.0, weight="bold", color="#1a237e")
    info_text = "EN3160 Computer Vision | Jaindu Samaranayaka (Index: 230563H) | University of Moratuwa"
    if subtitle:
        info_text = f"{subtitle}  |  {info_text}"
    header_ax.text(0.0, 0.15, info_text, fontsize=8.2, color="#37474f", style="italic")
    header_ax.plot([0, 1], [0.0, 0.0], color="#1a237e", linewidth=1.5, transform=header_ax.transAxes)


def add_footer(fig, page_num: int, total_pages: int = 4):
    """Add footer with page number."""
    footer_ax = fig.add_axes([0.06, 0.015, 0.88, 0.025])
    footer_ax.axis("off")
    footer_ax.plot([0, 1], [0.85, 0.85], color="#b0bec5", linewidth=0.8, transform=footer_ax.transAxes)
    footer_ax.text(0.5, 0.20, f"Page {page_num} of {total_pages}", ha="center", fontsize=8.2, color="#546e7a")


def generate_page_1():
    """Page 1: Question 1 - Sunflower Center Detection via Scale-Space LoG."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    add_header(fig, "Task 1: Sunflower Center Detection with Scale-Space LoG Extrema", "Assignment 2 Report")

    # Section 1: Overview & Theory (concise to fit nicely above figure and table)
    text_ax = fig.add_axes([0.06, 0.74, 0.88, 0.17])
    text_ax.axis("off")
    p1_lines = [
        r"1. Theoretical Framework & Scale-Space LoG Blob Detection",
        r"Blob detection identifies circular image structures by finding scale-space extrema of the scale-normalized",
        r"Laplacian of Gaussian:  $\nabla^2_{\mathrm{norm}} (x, y; \sigma) = \sigma^2 \nabla^2 [G(x, y; \sigma) * I(x, y)]$. The scale factor $\sigma^2$ compensates",
        r"for Gaussian attenuation, ensuring scale invariance. Sunflower flower heads feature dark central floret disks",
        r"encircled by bright yellow petals. For a dark blob on a bright background, the Laplacian second derivatives are",
        r"positive ($\nabla^2 I > 0$); thus, the positive normalized LoG ($+\sigma^2 \nabla^2 [G_\sigma * I]$) peaks at the dark disk center",
        r"with characteristic radius $r = \sqrt{2}\sigma$. An HSV petal envelope gates candidate regions, suppressing foliage shadows.",
        r"• Scale Sampling: 24 geometrically spaced scales from $\sigma_{\mathrm{min}} = 1.50$ px to $\sigma_{\mathrm{max}} = 12.50$ px ($r \in [2.12, 17.68]$ px).",
        r"• 3D Extrema & NMS: Scale-space local maxima above threshold (0.035) are filtered via greedy spatial NMS,",
        r"  ensuring exactly one fitted circle per sunflower disk and suppressing petal boundary artifacts.",
    ]
    text_ax.text(0, 1.0, "\n".join(p1_lines), va="top", fontsize=8.1, linespacing=1.26)

    # Image Result
    img_path = RESULTS_DIR / "question_1" / "sunflower_log_detections.png"
    if img_path.exists():
        im = mpimg.imread(str(img_path))
        im_ax = fig.add_axes([0.06, 0.35, 0.44, 0.37])
        im_ax.imshow(im)
        im_ax.axis("off")
        im_ax.set_title("Figure 1.1: Detected Sunflowers (50 circles)", fontsize=9.0, weight="bold", pad=4)

    # Table of Largest Detections
    tbl_ax = fig.add_axes([0.53, 0.35, 0.41, 0.37])
    tbl_ax.axis("off")
    tbl_ax.set_title("Table 1.1: Top 10 Largest Detected Circles", fontsize=9.0, weight="bold", pad=4)
    
    table_data = [
        ["Rank", "Center (x, y)", "Scale $\sigma$", "Radius $r$", "LoG Response"],
        ["1", "(282, 338)", "7.19 px", "10.17 px", "0.3499"],
        ["2", "(0, 275)", "7.19 px", "10.17 px", "0.3455"],
        ["3", "(106, 255)", "5.98 px", "8.46 px", "0.3630"],
        ["4", "(179, 327)", "5.98 px", "8.46 px", "0.3451"],
        ["5", "(178, 261)", "5.45 px", "7.71 px", "0.3511"],
        ["6", "(189, 240)", "4.53 px", "6.41 px", "0.3628"],
        ["7", "(63, 243)", "4.53 px", "6.41 px", "0.3551"],
        ["8", "(274, 258)", "4.53 px", "6.41 px", "0.3448"],
        ["9", "(19, 251)", "4.53 px", "6.41 px", "0.2912"],
        ["10", "(161, 290)", "4.14 px", "5.85 px", "0.3618"],
    ]
    table = tbl_ax.table(cellText=table_data, colWidths=[0.12, 0.28, 0.20, 0.20, 0.22], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.45)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#e8eaf6")
        table[(0, i)].set_text_props(weight="bold", color="#1a237e")

    # Code Box & Analysis
    code_ax = fig.add_axes([0.06, 0.05, 0.88, 0.27])
    code_ax.axis("off")
    code_lines = [
        "Core Implementation Snippet (Dark-Blob Scale-Space LoG & Single-Circle NMS):",
        "  sigmas = np.geomspace(1.5, 12.5, 24)",
        "  responses = [+(s**2) * cv2.Laplacian(cv2.GaussianBlur(gray_float, (0,0), s, s), cv2.CV_32F) for s in sigmas]",
        "  scale_space = np.stack(responses, axis=0)",
        "  local_max = scale_space == maximum_filter(scale_space, size=(3, 5, 5), mode='nearest')",
        "  # Disk candidate filtering: within yellow petal envelope, dark intensity, y >= 0.51*H",
        "  candidates = [(scale_space[k,y,x], x, y, s, np.sqrt(2)*s) for k,y,x in pts if valid_sunflower(x,y)]",
        "",
        "Analysis of Scale-Space Extrema Across Depth:",
        "• Near Foreground Flowers: Exhibit radii r = 7.7 - 10.2 px (sigma = 5.5 - 7.2 px), perfectly fitting the prominent disks.",
        "• Midground and Background Flowers: Smoothly scale down (r = 2.1 - 6.4 px, sigma = 1.5 - 4.5 px) as distance increases.",
        "• Single Circle per Flower: Overlap-aware spatial NMS completely suppresses spurious petal ring detections.",
    ]
    code_ax.text(0, 1.0, "\n".join(code_lines), va="top", fontsize=8.0, family="monospace", linespacing=1.24)

    add_footer(fig, 1)
    return fig


def generate_page_2():
    """Page 2: Question 2 - Robust Line and Circle Fitting using RANSAC."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    add_header(fig, "Task 2: Robust Line and Circle Fitting with RANSAC", "Assignment 2 Report")

    # Section 1: Overview & Equations
    text_ax = fig.add_axes([0.06, 0.74, 0.88, 0.17])
    text_ax.axis("off")
    p2_lines = [
        r"1. Mathematical Formulation & Sequential RANSAC Estimation",
        r"Given a contaminated point cloud $X = X_{\mathrm{circ}} \cup X_{\mathrm{line}}$ ($N=100$ points) generated from Listing 1 of the assignment brief,",
        r"sequential RANSAC is applied to estimate model parameters under high noise and contamination:",
        r"• Line Parameterization: Unit normal $[a, b]^T$ with constraint $\|[a, b]\| = 1$ and perpendicular offset $d$, satisfying",
        r"  $a x + b y = d$. Orthogonal algebraic error is $e_i = |a x_i + b y_i - d|$. From each 2-point sample, the unit normal is computed.",
        r"  Consensus points ($e_i < 1.0$) are refitted via Total Least Squares (SVD on zero-mean consensus points, right singular vector).",
        r"• Circle Parameterization: Center $[x_0, y_0]^T$ and radius $r$. Candidate triplets yield circumcenters via linear 2x2 systems.",
        r"  Radial error is $e_i = \left| \| [x_i, y_i]^T - [x_0, y_0]^T \|_2 - r \right|$. Consensus ($e_i < 0.8$) is refitted via non-linear least squares.",
    ]
    text_ax.text(0, 1.0, "\n".join(p2_lines), va="top", fontsize=8.2, linespacing=1.28)

    # Unified Figure matching Figure 1 of brief
    img_path = RESULTS_DIR / "question_2" / "line_circle_ransac.png"
    if img_path.exists():
        im = mpimg.imread(str(img_path))
        im_ax = fig.add_axes([0.06, 0.32, 0.44, 0.39])
        im_ax.imshow(im)
        im_ax.axis("off")
        im_ax.set_title("Figure 2.1: Unified RANSAC Line & Circle Fits", fontsize=9.0, weight="bold", pad=4)

    # Comparison Table
    tbl_ax = fig.add_axes([0.53, 0.54, 0.41, 0.17])
    tbl_ax.axis("off")
    tbl_ax.set_title("Table 2.1: Ground Truth vs. Estimated Models", fontsize=9.0, weight="bold", pad=4)
    table_data = [
        ["Parameter", "Ground Truth", "RANSAC Fit", "Error"],
        ["Normal $[a, b]^T$", "[0.7071, 0.7071]", "[0.7044, 0.7098]", r"$\Delta \mathbf{n} = 0.0038$"],
        ["Distance $d$", "1.4142", "1.6010", r"$\Delta d = 0.1868$"],
        ["Center $(x_0, y_0)$", "(2.000, 3.000)", "(2.186, 2.739)", r"$\Delta c = 0.320$"],
        ["Radius $r$", "10.000", "9.959", r"$\Delta r = 0.041$"],
        ["Line Inliers", "50", "49 / 100", "98% Recall"],
        ["Circle Inliers", "50", "39 / 51 remnant", "78% Recall"],
    ]
    table = tbl_ax.table(cellText=table_data, colWidths=[0.30, 0.25, 0.25, 0.20], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.35)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#e8eaf6")
        table[(0, i)].set_text_props(weight="bold", color="#1a237e")

    # Discussion of 2(d)
    disc_ax = fig.add_axes([0.53, 0.32, 0.41, 0.19])
    disc_ax.axis("off")
    disc_ax.set_title("Question 2(d) Analysis: Fitting Order Effect", fontsize=9.0, weight="bold", pad=4)
    disc_lines = [
        "What happens if we fit the circle first?",
        "• Degeneracy Traps: 3 nearly collinear points sampled from the line",
        r"  produce an enormous circle ($r \to \infty$). Such shallow arcs capture",
        "  line points into their consensus, corrupting circle parameters.",
        "• Sample Complexity: A line requires s=2 points; a circle requires s=3.",
        r"  Uncontaminated sample probability is $P = w^s$. For $w=0.5$:",
        r"  $P_{\mathrm{line}} = 0.5^2 = 0.25$;  $P_{\mathrm{circ}} = 0.5^3 = 0.125$.",
        "  Line RANSAC converges twice as reliably.",
        "• Conclusion: Fitting the lower-parameter model (line) first cleans",
        "  the remnant set, ensuring circle optimization stability.",
    ]
    disc_ax.text(0, 1.0, "\n".join(disc_lines), va="top", fontsize=7.8, linespacing=1.22)

    # Code Box
    code_ax = fig.add_axes([0.06, 0.05, 0.88, 0.24])
    code_ax.axis("off")
    code_lines = [
        "Key Implementation Snippet (TLS Line Refit & Radial Error Circle RANSAC):",
        "  # 1. Total Least Squares Line Refit via SVD:",
        "  centroid = inlier_points.mean(axis=0); _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)",
        "  normal = vh[-1] / np.linalg.norm(vh[-1]); d = float(normal @ centroid)",
        "  # 2. Circle Circumcenter & Residual Optimization on Remnant Points:",
        "  def circle_from_three_points(p1, p2, p3):",
        "      M = 2.0 * np.vstack((p2 - p1, p3 - p1)); rhs = np.array([p2@p2 - p1@p1, p3@p3 - p1@p1])",
        "      center = np.linalg.solve(M, rhs); return center[0], center[1], np.linalg.norm(center - p1)",
        "  fit = least_squares(lambda p: np.linalg.norm(remnant_inliers - p[:2], axis=1) - p[2], x0=best_model)",
    ]
    code_ax.text(0, 1.0, "\n".join(code_lines), va="top", fontsize=8.0, family="monospace", linespacing=1.24)

    add_footer(fig, 2)
    return fig


def generate_page_3():
    """Page 3: Question 3 - Planar Flag Superimposition via Homography Warping."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    add_header(fig, "Task 3: Planar Flag Warping & Blending via Homography", "Assignment 2 Report")

    # Section 1: Homography & Bug Resolution
    text_ax = fig.add_axes([0.06, 0.74, 0.88, 0.17])
    text_ax.axis("off")
    p3_lines = [
        r"1. Projective Homography Formulation & IDE Interactive Corner Selection Resolution",
        r"A planar homography is an invertible projective mapping $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ defined up to scale (8 degrees of freedom):",
        r"$\mathbf{x}' \sim \mathbf{H}\mathbf{x}$. Each 2D point correspondence provides 2 independent linear constraints; a minimum of 4 non-collinear",
        r"correspondences uniquely solves for H using Direct Linear Transformation (DLT) via Singular Value Decomposition (SVD).",
        r"Investigation of Live Interactive Corner Selection Issue in IDE:",
        r"• Root Cause Analysis: (1) In IDE notebook environments (VS Code / web notebooks), default matplotlib rendering (`inline` or `Agg`)",
        r"  produces static raster outputs, disabling GUI event dispatchers (`mpl_connect`); (2) Synchronous calls (`plt.ginput`) freeze the kernel",
        r"  indefinitely waiting for desktop window events; (3) Starter image `flag.png` contains 3 channels (BGR), but starter code checked",
        r"  `shape[2]==4`, throwing unhandled `ValueError` before any corner selection could execute.",
        r"• Comprehensive Resolution: Designed a dual-mode system in `src/corner_selector.py`: (a) A native Win32 OpenCV GUI window",
        r"  with mouse callbacks, crosshairs, coordinate readouts, and undo/reset keys; (b) Calibrated default presets for automated / IDE runs.",
    ]
    text_ax.text(0, 1.0, "\n".join(p3_lines), va="top", fontsize=8.0, linespacing=1.24)

    # Figures: Corner overlay and 3 placements
    # Overlay
    img1_path = RESULTS_DIR / "question_3" / "wadham_corners_overlay.png"
    if img1_path.exists():
        im1 = mpimg.imread(str(img1_path))
        ax1 = fig.add_axes([0.06, 0.44, 0.43, 0.28])
        ax1.imshow(im1)
        ax1.axis("off")
        ax1.set_title("Figure 3.1: Planar Placement Boundaries", fontsize=8.6, weight="bold")

    # Placement 1 (Figure 2 replication)
    img2_path = RESULTS_DIR / "question_3" / "wadham_flag_1.png"
    if img2_path.exists():
        im2 = mpimg.imread(str(img2_path))
        ax2 = fig.add_axes([0.51, 0.44, 0.43, 0.28])
        ax2.imshow(im2)
        ax2.axis("off")
        ax2.set_title("Figure 3.2: Placement 1 (Courtyard Wall - Fig. 2 Replication)", fontsize=8.6, weight="bold")

    # Placement 2 (Lawn display)
    img3_path = RESULTS_DIR / "question_3" / "wadham_flag_2.png"
    if img3_path.exists():
        im3 = mpimg.imread(str(img3_path))
        ax3 = fig.add_axes([0.06, 0.17, 0.43, 0.25])
        ax3.imshow(im3)
        ax3.axis("off")
        ax3.set_title("Figure 3.3: Placement 2 (Courtyard Lawn Display)", fontsize=8.6, weight="bold")

    # Placement 3 (Right facade)
    img4_path = RESULTS_DIR / "question_3" / "wadham_flag_3.png"
    if img4_path.exists():
        im4 = mpimg.imread(str(img4_path))
        ax4 = fig.add_axes([0.51, 0.17, 0.43, 0.25])
        ax4.imshow(im4)
        ax4.axis("off")
        ax4.set_title("Figure 3.4: Placement 3 (Chapel Wall Facade)", fontsize=8.6, weight="bold")

    # Non-technical Rationale & Discussion
    rationale_ax = fig.add_axes([0.06, 0.04, 0.88, 0.11])
    rationale_ax.axis("off")
    rat_lines = [
        "Non-Technical Rationale for Image & Placement Choices:",
        "• Placement 1 (Left Courtyard Wall Facade): Accurately replicates Figure 2 from the assignment brief, superimposing the British flag onto",
        "  the residential quadrangle of Oxford's historic Wadham College (founded 1610). Vertical masonry buttresses define a planar surface.",
        "• Placement 2 (Courtyard Lawn): Places the flag horizontally across the manicured lawn, simulating a ceremonial event carpet for matriculation",
        "  or university garden parties. This illustrates the homography's ability to handle steep ground-plane projective foreshortening.",
        "• Placement 3 (Right Chapel Wall): Projects the flag onto the adjacent perpendicular facade, demonstrating multi-planar perspective convergence.",
    ]
    rationale_ax.text(0, 1.0, "\n".join(rat_lines), va="top", fontsize=7.8, linespacing=1.22)

    add_footer(fig, 3)
    return fig


def generate_page_4():
    """Page 4: Question 4 - PCB Alignment and Defect Inspection via SIFT & Homography."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    add_header(fig, "Task 4: PCB Alignment & Difference Inspection via SIFT", "Assignment 2 Report")

    # Section 1: Overview
    text_ax = fig.add_axes([0.06, 0.77, 0.88, 0.14])
    text_ax.axis("off")
    p4_lines = [
        "1. SIFT Feature Matching, RANSAC Alignment, and Differential Inspection",
        "The objective is to align test PCB boards against reference template `AECC1.jpg` using planar homographies and detect differences.",
        "• SIFT Feature Detection & Matching: SIFT detects scale/rotation-invariant keypoints (`cv2.SIFT_create()`). Putative correspondences are",
        "  established via brute-force L2 matching with Lowe's ratio test (d1 < 0.75 d2). Homography H is estimated with RANSAC (5.0 px threshold).",
        "• Warping & Absolute Difference: The test PCB is warped via H to template dimensions (1800x1200). The absolute grayscale difference",
        "  D(x, y) = |I_template(x, y) - I_aligned(x, y)| highlights physical manufacturing discrepancies and assembly variations.",
    ]
    text_ax.text(0, 1.0, "\n".join(p4_lines), va="top", fontsize=8.0, linespacing=1.25)

    # Table of Results across all PCBs
    tbl_ax = fig.add_axes([0.06, 0.62, 0.88, 0.14])
    tbl_ax.axis("off")
    tbl_ax.set_title("Table 4.1: SIFT Matching and RANSAC Alignment Statistics across All PCB Variants", fontsize=8.8, weight="bold", pad=3)
    table_data = [
        ["PCB Variant", "Template Keypts", "PCB Keypts", "Ratio Matches", "RANSAC Inliers", "Inlier Ratio", "Status"],
        ["AECC1 (Template)", "7,963", "-", "-", "-", "-", "Reference Baseline"],
        ["AECC2", "7,963", "8,272", "4,606", "3,951", "85.8%", "Aligned (Minor Diff)"],
        ["AECC3", "7,963", "9,890", "5,652", "4,997", "88.4%", "Aligned (Minor Diff)"],
        ["AECC4", "7,963", "7,381", "4,009", "3,424", "85.4%", "Aligned (Minor Diff)"],
        ["ADAD1", "7,963", "2,986", "655", "226", "34.5%", "Aligned (Component Shifts)"],
        ["AAAA1", "7,963", "1,873", "395", "8", "2.0%", "Weak (Different Family)"],
    ]
    table = tbl_ax.table(cellText=table_data, colWidths=[0.20, 0.15, 0.13, 0.13, 0.13, 0.12, 0.20], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#e8eaf6")
        table[(0, i)].set_text_props(weight="bold", color="#1a237e")

    # Images for AECC2
    img1_path = RESULTS_DIR / "question_4" / "AECC2_matches.png"
    img2_path = RESULTS_DIR / "question_4" / "AECC2_difference.png"
    if img1_path.exists() and img2_path.exists():
        im1 = mpimg.imread(str(img1_path))
        im2 = mpimg.imread(str(img2_path))
        
        ax1 = fig.add_axes([0.06, 0.38, 0.54, 0.22])
        ax1.imshow(im1)
        ax1.axis("off")
        ax1.set_title("Figure 4.1: AECC2 RANSAC Matches (3,951 inliers)", fontsize=8.4, weight="bold")

        ax2 = fig.add_axes([0.62, 0.38, 0.32, 0.22])
        ax2.imshow(im2, cmap="gray")
        ax2.axis("off")
        ax2.set_title("Figure 4.2: AECC2 Difference Map", fontsize=8.4, weight="bold")

    # Images for ADAD1
    img3_path = RESULTS_DIR / "question_4" / "ADAD1_difference.png"
    if img3_path.exists():
        im3 = mpimg.imread(str(img3_path))
        ax3 = fig.add_axes([0.06, 0.17, 0.43, 0.18])
        ax3.imshow(im3, cmap="gray")
        ax3.axis("off")
        ax3.set_title("Figure 4.3: ADAD1 Difference (Component Shifts)", fontsize=8.4, weight="bold")

    # Images for AAAA1
    img4_path = RESULTS_DIR / "question_4" / "AAAA1_difference.png"
    if img4_path.exists():
        im4 = mpimg.imread(str(img4_path))
        ax4 = fig.add_axes([0.51, 0.17, 0.43, 0.18])
        ax4.imshow(im4, cmap="gray")
        ax4.axis("off")
        ax4.set_title("Figure 4.4: AAAA1 Difference (Distinct Family)", fontsize=8.4, weight="bold")

    # Section 2: Answer to Question 4(e)
    ans_ax = fig.add_axes([0.06, 0.04, 0.88, 0.11])
    ans_ax.axis("off")
    ans_lines = [
        "Answer to Question 4(e): Root Causes of High Difference Values in Specific Components:",
        "1. Physical Variance: Across board revisions (AECC vs. ADAD vs. AAAA), manufacturers substitute component packages (e.g. through-hole vs.",
        "   SMD capacitors, gold vs. silver USB shell plating, differing manufacturer branding silkscreens, or relocated reset pushbuttons).",
        "2. 3D Parallax vs. 2D Planar Homography: Homography assumes that all scene points lie strictly on a 2D planar surface.",
        "   However, tall components (e.g. the 11mm tall USB Type B port and cylindrical capacitors) exhibit out-of-plane height. Under changing camera viewpoints,",
        "   tall structures undergo 3D projective parallax that cannot be geometrically registered by a single 2D planar homography.",
        "3. Specular Reflection: Metallic connector shields and solder joints cause viewpoint-dependent specular glints, creating localized intensity deltas.",
    ]
    ans_ax.text(0, 1.0, "\n".join(ans_lines), va="top", fontsize=7.8, linespacing=1.22)

    add_footer(fig, 4)
    return fig


def main():
    print("Generating comprehensive 4-page PDF report...")
    pdf = PdfPages(str(PDF_PATH))

    fig1 = generate_page_1()
    pdf.savefig(fig1, dpi=200)
    plt.close(fig1)
    print("Page 1 generated.")

    fig2 = generate_page_2()
    pdf.savefig(fig2, dpi=200)
    plt.close(fig2)
    print("Page 2 generated.")

    fig3 = generate_page_3()
    pdf.savefig(fig3, dpi=200)
    plt.close(fig3)
    print("Page 3 generated.")

    fig4 = generate_page_4()
    pdf.savefig(fig4, dpi=200)
    plt.close(fig4)
    print("Page 4 generated.")

    pdf.close()
    print(f"Report saved successfully to: {PDF_PATH}")


if __name__ == "__main__":
    main()
