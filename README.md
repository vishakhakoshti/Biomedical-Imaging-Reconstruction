# Biomedical Imaging Reconstruction

A complete CT image reconstruction pipeline implemented in **Python** as part of the Biomedical Engineering coursework.  
This project builds a full workflow from **image grid creation** to **GPU-accelerated backprojection**, demonstrating the principles of medical image reconstruction.

---

## 📚 Project Overview

| Task | Topic | Description |
|------|--------|-------------|
| **Task 1** | Grid Class & Phantom | Implemented a 2D grid class with physical coordinates and interpolation; visualized the *Modified Shepp–Logan Phantom*. |
| **Task 2** | Parallel-Beam FBP | Generated a sinogram using the Radon transform, applied **Ramp** and **Ram-Lak** filters, and performed filtered backprojection. |
| **Task 3** | Fan-Beam FBP | Extended the model to **fan-beam geometry**, created fanograms, performed **rebinning** and **cosine-weighted backprojection**. |
| **Task 4** | OpenCL Acceleration | Implemented **GPU-based grid addition** and **parallel-beam backprojection** using **PyOpenCL**, comparing CPU vs GPU performance. |

---

## ⚙️ Requirements

**Python ≥ 3.7**

Install dependencies:
```bash
pip install numpy matplotlib scipy pyopencl

# How to Run
Each task has its own folder with runnable scripts.

## Task 1 – Grid & Phantom
cd Task1_Grid
python SheppLoganShow.py

# Task 2 – Parallel-Beam Reconstruction
cd Task2_ParallelBeam
python Ex2Run.py

# Task 3 – Fan-Beam Reconstruction
cd Task3_FanBeam
python Ex3Run.py

# Task 4 – GPU / OpenCL
cd Task4_OpenCL
python Ex4Run1.py   # Grid addition on GPU
python Ex4Run2.py   # GPU-accelerated backprojection

# Folder Structure
Biomedical-Imaging-Reconstruction/
│
├── Task1_Grid/
├── Task2_ParallelBeam/
├── Task3_FanBeam/
├── Task4_OpenCL/
└── README.md

Each folder includes:

Exercise #.pdf – Original instructions

Grid.py, Methods.py, interpolate.py, phantom.py – Core modules

Ex#Run.py – Main runnable scripts
