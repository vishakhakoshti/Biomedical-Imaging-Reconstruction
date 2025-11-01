# 🧬 Biomedical Imaging Reconstruction

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
