# EN3160 Neural Networks — Assignment 2

This repository is the starter structure for the EN3160 assignment on fitting
and alignment. The assignment brief is available in
[`en3160_assignment_02.pdf`](./en3160_assignment_02.pdf).

## Assignment sections

1. Detect circles in the sunflower-field image using Laplacian of Gaussians
   and scale-space extrema.
2. Implement RANSAC line and circle fitting for a noisy point set.
3. Warp flag images onto planar surfaces using homographies.
4. Align the PCB images to `AECC1.jpg` and compare the results.

This commit only sets up the repository; the assignment solutions have not
been implemented.

## Repository layout

```text
.
├── a2images/                 # Supplied images, including PCB images
├── report/                   # Final report (maximum four pages)
├── results/
│   ├── question_1/
│   ├── question_2/
│   ├── question_3/
│   └── question_4/
├── src/                      # Assignment implementation
├── .gitignore
├── en3160_assignment_02.pdf
└── requirements.txt
```

## Setup

Use Python 3 and install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Keep generated figures and intermediate outputs in the corresponding
`results/question_N/` folder. Save the final report in `report/` using the
filename format required by the assignment.
