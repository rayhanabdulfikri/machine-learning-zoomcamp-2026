# Machine Learning Zoomcamp 2026 — Homework 01

> **From raw car data to linear regression — my first step into Machine Learning.**

This repository contains my work for **Machine Learning Zoomcamp 2026 — Homework 01**, where I explored the fundamentals of data analysis with **Pandas and NumPy** and implemented a simple **linear regression model from scratch using matrix operations**.

The project starts with basic dataset exploration and gradually moves toward a mathematical machine learning implementation.

---

## What I Built

The notebook covers the following workflow:

```text
Car Dataset
    │
    ├── Explore the data
    │
    ├── Check dataset size
    │
    ├── Explore fuel types
    │
    ├── Find missing values
    │
    ├── Analyze Asian cars
    │
    ├── Handle missing horsepower values
    │
    └── Implement Linear Regression
            │
            ├── XᵀX
            ├── (XᵀX)⁻¹
            ├── w = (XᵀX)⁻¹Xᵀy
            └── Interactive 3D Regression
```

---

## Dataset

The project uses the **Car Fuel Efficiency 2026** dataset.

Some of the variables used in the analysis include:

| Feature               | Description       |
| --------------------- | ----------------- |
| `origin`              | Vehicle origin    |
| `fuel_type`           | Type of fuel      |
| `vehicle_weight`      | Vehicle weight    |
| `model_year`          | Model year        |
| `horsepower`          | Engine horsepower |
| `fuel_efficiency_mpg` | Fuel efficiency   |

---

## Key Exercises

### 1. Data Exploration

I started by inspecting the dataset and answering basic questions such as:

* What version of Pandas is being used?
* How many records are in the dataset?
* How many fuel types are available?
* Which columns contain missing values?

This establishes the basic workflow for understanding an unfamiliar dataset before modeling.

### 2. Filtering and Feature Analysis

For several exercises, I filtered the dataset to focus specifically on **cars from Asia**.

Example:

```python
asia = df[df["origin"] == "Asia"]
```

I then explored variables such as `fuel_efficiency_mpg`, `vehicle_weight`, and `model_year`.

### 3. Missing Value Handling

The `horsepower` column contains missing values.

I calculated:

```text
Median → Mode → Fill missing values → Median again
```

The missing values were filled using the most frequent horsepower value:

```python
mode_horsepower = df["horsepower"].mode()[0]

df["horsepower"] = df["horsepower"].fillna(mode_horsepower)
```

The median was then recalculated to determine whether the imputation changed its value.

---

# Linear Regression From Scratch

The final exercise goes beyond using a pre-built machine learning library.

Instead, I implemented the regression calculation directly with **NumPy matrix operations**.

The features are:

```text
vehicle_weight
model_year
```

The target values are:

```text
[1100, 1300, 800, 900, 1000, 1100, 1200]
```

The workflow is based on the normal equation:

$$
w = (X^TX)^{-1}X^Ty
$$

The implementation follows these steps:

```python
XTX = X.T @ X

XTX_inv = np.linalg.inv(XTX)

w = XTX_inv @ X.T @ y
```

Finally:

```python
round(w.sum(), 3)
```

This exercise helped me understand that linear regression is not just a black-box function. The model can be expressed directly through **linear algebra and matrix operations**.

---

# Interactive 3D Visualization

To make the regression easier to understand, I added an interactive **3D visualization using Plotly**.

The visualization contains:

* **Actual data points**
* `vehicle_weight` on the X-axis
* `model_year` on the Y-axis
* `y` on the Z-axis
* A **regression plane** generated from the calculated weights

The visualization can be rotated, zoomed, and explored interactively.

## Interactive 3D Regression

Explore the regression model interactively:

[Open Interactive 3D Regression](https://rayhanabdulfikri.github.io/machine-learning-zoomcamp-2026/01_Introduction/regression_3d.html)

---

# What I Learned

Through this homework, I practiced several important foundations:

```text
✓ Pandas DataFrame manipulation
✓ Filtering and selecting data
✓ Handling missing values
✓ NumPy arrays
✓ Matrix multiplication
✓ Matrix inversion
✓ Linear algebra for Machine Learning
✓ Normal equation
✓ Interactive data visualization
```

More importantly, the final exercise connected three concepts together:

```text
Data
  ↓
Linear Algebra
  ↓
Machine Learning
```

This is one of the first steps toward understanding how machine learning models work internally rather than only using high-level libraries.

---

# Tools

```text
Python
Pandas
NumPy
Plotly
Jupyter Notebook
```

---

# Project Structure

```text
.
├── HW_01_RayhanAbdulFikri(1).ipynb
├── car_fuel_efficiency_2026.csv
├── regression_3d.html
└── introduction.md
```

---

# Course

**Machine Learning Zoomcamp 2026**

Homework 01:

[View the original homework](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01)

---

## Author

**Rayhan Abdul Fikri**

[GitHub](https://github.com/rayhanabdulfikri/)
