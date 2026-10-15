# EN3160 Fitting and Alignment — Assignment 2

Repository for EN3160 Computer Vision Assignment 2 on Fitting and Alignment.
- **Student**: Jaindu Samaranayaka
- **Index**: 230563H
- **Assignment Brief**: [`en3160_assignment_02.pdf`](./en3160_assignment_02.pdf)
- **Final Report**: [`report/230563H_a02.pdf`](./report/230563H_a02.pdf) (4 pages)

---

## Assignment Tasks

1. **Question 1: Sunflower Center Detection via Scale-Space LoG**
   - Detects dark central disks using positive scale-normalized Laplacian of Gaussian: $+\sigma^2 \nabla^2(G_\sigma * I)$.
   - Employs yellow-petal HSV envelope gating and spatial non-maximum suppression (NMS) to fit exactly one circle ($r = \sqrt{2}\sigma$) per flower disk across depth.
   - Script: [`src/q1_sunflower_detector.py`](./src/q1_sunflower_detector.py) | Result: [`results/question_1/sunflower_log_detections.png`](./results/question_1/sunflower_log_detections.png)

2. **Question 2: Robust Line and Circle Fitting via Sequential RANSAC**
   - Implements custom RANSAC with unit normal parameterization $\|[a, b]^\top\| = 1$.
   - Sequentially estimates the line, removes inliers, and fits the remnant circle using radial error consensus and Nelder-Mead geometric distance minimization.
   - Analyzes sensitivity and explains why fitting circle first captures spurious tangential arcs.
   - Script: [`src/q2_ransac_fitting.py`](./src/q2_ransac_fitting.py) | Result: [`results/question_2/line_circle_ransac.png`](./results/question_2/line_circle_ransac.png)

3. **Question 3: Planar Homography Warping and Flag Blending**
   - Dual-mode corner selector ([`src/corner_selector.py`](./src/corner_selector.py)): supports live OpenCV interactive GUI clicking as well as robust calibrated presets for headless/IDE execution.
   - Computes Direct Linear Transformation (DLT) via SVD and warps the flag onto three distinct architectural planes with alpha blending:
     1. Left Courtyard Wall (Assignment Figure 2 replication)
     2. Courtyard Lawn (Ceremonial banner display)
     3. Right Chapel Facade (Orthogonal plane)
   - Script: [`src/q3_homography_warper.py`](./src/q3_homography_warper.py) | Results: [`results/question_3/`](./results/question_3/)

4. **Question 4: PCB SIFT Matching & Homography Alignment**
   - Detects SIFT keypoints, matches with BFMatcher and Lowe's ratio test (0.75).
   - Computes robust homography with RANSAC to align all 5 test PCBs (`AECC2` - `AECC6`) to template `AECC1`.
   - Computes absolute difference images highlighting missing components, solder bridges, and assembly defects.
   - Script: [`src/q4_pcb_alignment.py`](./src/q4_pcb_alignment.py) | Results: [`results/question_4/`](./results/question_4/)

---

## Repository Layout

```text
.
├── notebooks/
│   └── assignment_02.ipynb       # Complete executed Jupyter notebook with all outputs
├── report/
│   └── 230563H_a02.pdf           # Final 4-page submission report
├── results/
│   ├── question_1/               # Sunflower LoG circle detection plot
│   ├── question_2/               # Sequential RANSAC line & circle plots
│   ├── question_3/               # Flag warping across 3 architectural planes
│   └── question_4/               # SIFT matches, aligned PCBs, and difference images
├── src/
│   ├── corner_selector.py        # Interactive GUI + calibrated preset corner selector
│   ├── q1_sunflower_detector.py  # Question 1 standalone runner
│   ├── q2_ransac_fitting.py      # Question 2 standalone runner
│   ├── q3_homography_warper.py   # Question 3 standalone runner
│   └── q4_pcb_alignment.py       # Question 4 standalone runner
├── a2images/                     # Supplied input images (sunflower, Wadham, flag, PCBs)
├── generate_report.py            # PDF report generator script (builds 230563H_a02.pdf)
├── en3160_assignment_02.pdf      # Assignment brief
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation
```

---

## Setup & Running

1. **Install dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Run individual question scripts**:
   ```bash
   python src/q1_sunflower_detector.py
   python src/q2_ransac_fitting.py
   python src/q3_homography_warper.py
   python src/q4_pcb_alignment.py
   ```

3. **Regenerate report PDF**:
   ```bash
   python generate_report.py
   ```

4. **Launch Jupyter notebook**:
   ```bash
   jupyter notebook notebooks/assignment_02.ipynb
   ```
