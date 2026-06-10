# RSM Metric Matrix Design Configurator

## User Guide

### 1. Overview

This software is an optimization tool for **Response Surface Methodology (RSM)** designs. It constructs experimental matrices by combining **Hadamard**, **Conference**, and **Orthogonal Design (OD)** arrays to identify highly efficient experimental designs based on **D-optimality** and **Q-star** metrics.

The tool performs an automated search across multiple matrix configurations and evaluates each candidate design to determine the best-performing arrangements for second-order response surface models.

---

## 2. README

### Prerequisites

#### Python

* Python 3.x

#### Required Libraries

Install the required packages:

```bash
pip install numpy pandas scipy
```

#### Dependencies

The following file must be present in the same directory as the main script:

```text
generators.py
```

Failure to include this file will result in an import error during execution.

---

### How to Run

1. Open a terminal or command prompt.
2. Navigate to the project directory (P1 folder).
3. Execute the script:

```bash
python RMS_P1_v4.py
```

4. Follow the interactive prompts to configure your experiment.

---

### Input Parameters

#### Factors (m)

Number of experimental factors (input variables).

#### Hadamard Order

Select the order of the Hadamard matrix used as the base design.

#### Conference Matrix

Choose:

* `0` = No conference matrix
* `6–12` = Augment the base design using the selected conference matrix

#### Orthogonal Design (OD) Type

Select the Orthogonal Design structure to use:

```text
2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
```

---

## 3. Technical Documentation

### Core Logic

The design matrix construction follows the workflow below.

#### Step 1: Base Assembly

The initial design matrix, denoted by `Xf`, is constructed using:

* A Hadamard array
* Optionally augmented with a Conference matrix

This creates the starting experimental design.

#### Step 2: Dynamic Search

The `Construction3g2()` function performs a comprehensive grid search across all Orthogonal Design parameter combinations.

The search space consists of:

```math
2^{11}
```

possible configurations of the selected Orthogonal Design matrix (`XT`).

#### Step 3: Model Expansion

The `MakeModelMatrix()` function expands the design into a full second-order response surface model.

For a design containing `m` factors:

##### Original Design Space

```math
m
```

factors

##### Expanded Model Space

The resulting model matrix includes:

* 1 intercept term
* m linear effects
* m quadratic effects
* m(m−1)/2 two-factor interaction effects

Total model terms:

```math
1 + 2m + \frac{m(m-1)}{2}
```

#### Step 4: D-Efficiency Evaluation

The software computes the determinant of the information matrix:

```math
|X^T X|
```

This value is used to calculate the D-optimality criterion, allowing comparison of competing experimental designs.

Higher values indicate more statistically efficient designs.

---

### Metrics Explained

#### D-Value

A measure of design efficiency based on the determinant of the information matrix.

Characteristics:

* Larger values indicate better information content.
* Higher D-values correspond to more stable parameter estimation.
* Used as the primary optimization criterion.

#### Q-Star

A measure of design rotatability.

In the current implementation:

```text
Q* = 0.85
```

This value is fixed and serves as a placeholder rotatability metric.

---

## 4. Operational Instructions

### Post-Optimization Workflow

After the optimization search completes, the software performs the following sequence.

#### Console Summary

The best-performing designs for both:

* Evang Criterion
* Nguyen Criterion

are displayed in a summary table.

---

### Conditional Export

The program prompts:

```text
Do you want to create the Excel report and save the text files? (y/n)
```

#### If `n`

The program exits immediately.

No files are written to disk, preserving storage space.

#### If `y`

The software generates:

##### Excel Report

```text
F{f}_N{n}_H{h}_CM{cm}_OD{od}_Report.xlsx
```

##### Text Outputs

```text
Evang.txt
Nguyen.txt
```

These files contain the highest-performing design matrices identified during the optimization process.

---

## Troubleshooting

### ModuleNotFoundError: No module named 'numpy'

Install the required packages:

```bash
pip install numpy scipy pandas
```

If multiple Python versions are installed:

```bash
python -m pip install numpy scipy pandas
```

or

```bash
py -m pip install numpy scipy pandas
```

---

### ValueError: Dimension Mismatch

If a matrix dimension error occurs:

* Verify that the selected number of factors (`m`) is compatible with the chosen Orthogonal Design type.
* The software automatically slices `XA` to the first `m` columns.
* Very small factor counts combined with large Orthogonal Designs may require manual verification.

---

### Missing `generators.py`

If the file cannot be found, execution will terminate during import.

Verify that:

```text
generators.py
```

is located in the same directory as:

```text
RMS_P1_v4.py
```

---

## Output Summary

The software can generate:

| Output          | Description                               |
| --------------- | ----------------------------------------- |
| Console Summary | Displays optimal design metrics           |
| Excel Report    | Detailed optimization results             |
| Evang.txt       | Best design according to Evang criterion  |
| Nguyen.txt      | Best design according to Nguyen criterion |

---

## Notes

* Large Orthogonal Designs can produce lengthy execution times due to the exhaustive search strategy.
* Search complexity increases with both factor count and Orthogonal Design order.
* For best performance, ensure NumPy and SciPy are installed with optimized BLAS/LAPACK support.
