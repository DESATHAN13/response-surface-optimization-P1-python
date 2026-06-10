# Structural Matrix Search and Optimization Pipeline for RSM

A high-performance Python translation of the deterministic optimization and brute-force grid search pipeline used to construct highly efficient experimental designs for Response Surface Methodology (RSM). This repository contains the standalone script corresponding to the methodologies outlined in **Paper 01**.

The algorithm evaluates candidate multi-variable matrix frameworks over a multi-dimensional parameter space, analyzing structural layouts via dual criteria: statistical efficiency (**D-value measures** for parameter precision) and geometric uniformity (**Rotatability measures**).

---

## Technical Overview

The optimization pipeline operates through five critical computational phases:

1. **Parametric Orthogonal Design (OD) Generators:** Structural matrix-mapping blocks (ranging from $2 \times 2$ up to $20 \times 20$) accept scalar coordinate weights to force simultaneous factor transitions during axial runs.
2. **Factorial Block Assemblies:** Integrates a structured fractionated factorial foundation ($F$) utilizing a 32-run Hadamard base array combined with a specialized $10 \times 10$ conference matrix layout and its inverted reflection.
3. **Nested Search Engine Loops:** Executes an 11-dimensional nested coordinate loop (`itertools.product`) over binary coordinate shifts to build dynamic block-diagonal configurations.
4. **Full Second-Order Model Expansion:** Expands active composite candidate designs ($D = [X_f; T; C; -T]$) into a full second-order polynomial model matrix ($X_m$), adding distinct column arrays for intercepts, linear effects ($X_i$), pure quadratic effects ($X_i^2$), and two-factor cross-product interaction terms ($X_i X_j$).
5. **Criteria Evaluation & Validation:** Extracts the absolute determinant of the normalized information matrix product ($X_m' X_m$) against performance boundaries (**Evangelaras** and **Nguyen** criteria). High-performing layouts are instantly streamed directly to local logs.

---

## Complete Execution Instructions (via Google Chrome)

You can run this entire script, install dependencies, and generate data models inside your browser without installing anything locally by utilizing **Google Colab**.

### Phase 1: Running the Algorithm in the Cloud
1. Open Google Chrome and navigate to [Google Colab](https://colab.research.google.com/).
2. Sign in with your Google account and click **New Notebook**.
3. **Set up the Environment:** In the first code cell block, paste the following setup command and click the **Play** button on the left:
   ```bash
   !pip install numpy scipy
