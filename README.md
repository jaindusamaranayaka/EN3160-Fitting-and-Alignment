# EN3160 Fitting and Alignment — Assignment 2

Repository for EN3160 Computer Vision Assignment 2 on Fitting and Alignment.
- **Student**: Jaindu Samaranayaka
- **Index**: 230563H
- **GitHub Repository**: [jaindusamaranayaka/EN3160-Fitting-and-Alignment](https://github.com/jaindusamaranayaka/EN3160-Fitting-and-Alignment)
- **Assignment Brief**: [`en3160_assignment_02.pdf`](./en3160_assignment_02.pdf)
- **Final Report**: [`report/230563H_a02.pdf`](./report/230563H_a02.pdf) (4 pages)

---

## Assignment Tasks

1. **Question 1: Sunflower Center Detection via Scale-Space LoG Extrema**
   - Detects dark central floret disks using the positive scale-normalized Laplacian of Gaussian: $+\sigma^2 \nabla^2(G_\sigma * I)$ with characteristic radius $r = \sqrt{2}\sigma$.
   - Applies 4 hand-tuned candidate filters: (1) horizon row cutoff $y \ge 0.51H = 184$ px, (2) dilated HSV petal envelope ($H \in [14, 36], S, V \ge 70$), (3) grayscale dark floret test $I(x, y) \le 0.48$, and (4) LoG response peak $\ge 0.035$. Boundary margin $x, y \in [6, 354]$ px suppresses border reflections.
   - Evaluates perspective depth scaling trend ($r = 0.054y - 8.37$, $R = 0.87$) and documents scale boundary limitation at $\sigma_{\min} = 1.5$ px ($r = 2.12$ px).
   - Script: [`src/q1_sunflower_detector.py`](./src/q1_sunflower_detector.py) | Result: [`results/question_1/sunflower_log_detections.png`](./results/question_1/sunflower_log_detections.png)

2. **Question 2: Robust Line and Circle Fitting via Sequential RANSAC**
   - Implements custom RANSAC with unit normal parameterization $\|[a, b]^\top\| = 1$ and distance threshold $T = 0.85$.
   - Sequentially estimates the line, performs Total Least Squares (TLS) SVD refit on inliers, and fits the remnant circle using circumcenter sampling and Nelder-Mead optimization.
   - Features full numerical validation against ground truth ($\mathbf{n} \approx [0.7071, 0.7071]^\top, d \approx 1.4142, c=(2, 3), r=10$), plots both sample and refined fits, and justifies consensus sizes ($T_{\min} = 30-35$) and iterations using $N = \ln(1-p)/\ln(1-w^s)$.
   - Honestly runs and reports real circle-first RANSAC on the full point cloud: recovers circle with 45 inliers (39 circle points and 6 line points), demonstrating intersection contamination and fragmenting the remnant line.
   - Script: [`src/q2_ransac_fitting.py`](./src/q2_ransac_fitting.py) | Result: [`results/question_2/line_circle_ransac.png`](./results/question_2/line_circle_ransac.png)

3. **Question 3: Planar Homography Warping and Flag Blending**
   - Implements Direct Linear Transformation (DLT) via SVD and verifies against OpenCV's DLT solver to machine precision ($< 10^{-8}$).
   - Processes two distinct image pairs with plain-terms non-technical rationales:
     1. **Pair 1 (Wadham College + British Flag)**: Oxford residential quadrangle wall between dorm windows, celebrating British collegiate heritage.
     2. **Pair 2 (Civic Academic Building + Sri Lankan Flag)**: Main upper facade plane of a university hall for an international cultural festival celebrating student diversity.
   - Script: [`src/q3_homography_warper.py`](./src/q3_homography_warper.py) | Results: [`results/question_3/homography_warping_pairs.png`](./results/question_3/homography_warping_pairs.png)

4. **Question 4: PCB SIFT Matching, Homography Alignment, and Defect Inspection**
   - Detects SIFT keypoints, matches with BFMatcher and Lowe's ratio test (0.75), and estimates homography via RANSAC.
   - Delivers `cv2.drawMatches` correspondence visualization (4b) and comparative difference images across **all five** candidate boards (`AECC2`, `AECC3`, `AECC4`, `ADAD1`, `AAAA1`) (4d).
   - Code-generates alignment statistics table, classifies `AAAA1` as an **ALIGNMENT FAILURE** (< 20 inliers, 35.3%) without fabricating defect claims, identifies `ADAD1` layout mismatch, and localizes verified defect clusters on `AECC4`.
   - Explains causes of component differences in 4(e): 3D projective parallax, cast shadows, residual edge misregistration, specular glints, and component variations.
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

3. **Launch Jupyter notebook**:
   ```bash
   jupyter notebook notebooks/assignment_02.ipynb
   ```
