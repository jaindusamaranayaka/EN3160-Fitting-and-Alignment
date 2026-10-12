# EN3160 Fitting and Alignment — Assignment 2

Starter repository for the EN3160 computer vision assignment on fitting and
alignment. The assignment brief is [`en3160_assignment_02.pdf`](./en3160_assignment_02.pdf).

## Assignment tasks

1. Detect circles in the sunflower field image with Laplacian of Gaussian and
   scale-space extrema.
2. Fit lines and circles to noisy points with RANSAC.
3. Warp flag images onto planar surfaces using homographies.
4. Align PCB images to `AECC1.jpg` and compare the results.

This repository contains the assignment materials and project scaffold. Add
implementations under `src/`, keep each question's generated results in its
matching `results/question_N/` directory, and save the final report in `report/`
using the filename format specified in the assignment brief.

## Repository layout

```text
.
├── a2images/                 # Supplied images, including PCB images
├── report/                   # Final report
├── results/
│   ├── question_1/
│   ├── question_2/
│   ├── question_3/
│   └── question_4/
├── src/                      # Assignment implementation
├── en3160_assignment_02.pdf  # Assignment brief
├── requirements.txt          # Python dependencies
└── a2images.zip              # Supplied image archive
```

## Setup

Use Python 3 and install the required packages:

```bash
python -m pip install -r requirements.txt
```
