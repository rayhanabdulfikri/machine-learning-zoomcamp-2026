# Machine Learning Zoomcamp 2026 — Overview

A hands-on journey through **machine learning engineering**, covering the foundations of machine learning, classical models, evaluation, deployment, deep learning, serverless inference, Kubernetes, and capstone projects.

Course: [Machine Learning Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/)  
GitHub: [DataTalksClub / machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)

---

## Course Roadmap

The course progresses from core machine learning concepts to production-oriented ML engineering.

```text
Introduction to ML
        ↓
Regression
        ↓
Classification
        ↓
Evaluation Metrics
        ↓
Model Deployment
        ↓
Decision Trees & Ensemble Learning
        ↓
Neural Networks & Deep Learning
        ↓
Serverless Deep Learning
        ↓
Kubernetes
        ↓
Capstone Projects
```

---

## Homework

| # | Homework | Deadline | Status |
|---|---|---|---|
| 01 | [Introduction to Machine Learning](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01) | 29 September 2026, 06:00 | Open |
| 02 | [Machine Learning for Regression](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02) | 6 October 2026, 06:00 | Open |
| 03 | [Machine Learning for Classification](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03) | 13 October 2026, 06:00 | Open |
| 04 | [Evaluation Metrics for Classification](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04) | 20 October 2026, 06:00 | Open |
| 05 | [Deploying Machine Learning Models](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05) | 27 October 2026, 06:00 | Open |
| 06 | [Decision Trees and Ensemble Learning](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw06) | 3 November 2026, 06:00 | Open |
| 08 | [Neural Networks and Deep Learning](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw08) | 1 December 2026, 06:00 | Open |
| 09 | [Serverless Deep Learning](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw09) | 8 December 2026, 06:00 | Open |
| 10 | [Kubernetes](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw10) | 15 December 2026, 06:00 | Open |

> **Note:** The course overview currently lists Homework 01–06 and 08–10. Homework 07 is not listed in the provided course overview.

---

## Projects

| Project | Deadline | Status |
|---|---|---|
| Midterm Project | 17 November 2026, 06:00 | Closed |
| Capstone 1 | 5 January 2027, 06:00 | Closed |
| Capstone 2 | 19 January 2027, 06:00 | Closed |

---

## Learning Objectives

By the end of the course, the learning path is intended to cover:

- Machine learning fundamentals and problem framing
- Regression and classification
- Model evaluation and validation
- Feature engineering and model selection
- Tree-based models and ensemble methods
- Neural networks and deep learning
- Machine learning model deployment
- Serverless inference
- Kubernetes for ML workloads
- End-to-end machine learning projects

---

## Homework 01 — Introduction to Machine Learning

### Topics

Homework 01 focuses on the fundamentals of machine learning and a practical refresh of the Python data stack.

Key topics:

- Difference between machine learning and rule-based systems
- Supervised machine learning
- CRISP-DM
- Model selection
- Python environment setup
- NumPy
- Pandas
- Linear algebra
- Basic matrix operations
- Linear regression using the normal equation

### Dataset

**2026 Car Fuel Efficiency Dataset**

```text
https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

### Questions Covered

#### Q1 — Pandas version

Check the installed Pandas version:

```python
import pandas as pd

pd.__version__
```

#### Q2 — Records count

Determine the number of records in the dataset.

#### Q3 — Fuel types

Determine how many distinct fuel types are present.

Example:

```python
df["fuel_type"].unique()
```

#### Q4 — Missing values

Determine how many columns contain missing values.

```python
df.isna().sum()
```

#### Q5 — Maximum fuel efficiency for Asia

Filter the dataset to cars from Asia and find the maximum `fuel_efficiency_mpg`.

```python
asia = df[df["origin"] == "Asia"]

asia["fuel_efficiency_mpg"].max()
```

#### Q6 — Median horsepower

The task is to:

1. Calculate the median of `horsepower`.
2. Find its most frequent value (mode).
3. Fill missing `horsepower` values with the mode.
4. Calculate the median again.
5. Determine whether the median increased, decreased, or stayed the same.

Example:

```python
median_before = df["horsepower"].median()

mode_horsepower = df["horsepower"].mode()[0]

df["horsepower"] = df["horsepower"].fillna(mode_horsepower)

median_after = df["horsepower"].median()
```

#### Q7 — Linear regression matrix calculation

The task uses:

- Cars from Asia
- `vehicle_weight`
- `model_year`
- The first 7 rows
- A target vector `y`

The calculation implements the linear regression normal equation:

\[
w = (X^T X)^{-1} X^T y
\]

Example:

```python
import numpy as np

asia = df[df["origin"] == "Asia"]

X = asia[["vehicle_weight", "model_year"]]
X = X.head(7).values

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)

w = XTX_inv @ X.T @ y

print(round(float(w.sum()), 3))
```

---

## Repository Structure

A simple structure for keeping course work organized:

```text
machine-learning-zoomcamp-2026/
│
├── README.md
│
├── homework/
│   ├── 01-introduction-to-machine-learning/
│   │   ├── HW_01.ipynb
│   │   └── homework.py
│   │
│   ├── 02-regression/
│   ├── 03-classification/
│   ├── 04-evaluation-metrics/
│   ├── 05-deployment/
│   ├── 06-trees-ensemble/
│   ├── 08-neural-networks/
│   ├── 09-serverless/
│   └── 10-kubernetes/
│
└── projects/
    ├── midterm/
    ├── capstone-1/
    └── capstone-2/
```

---

## Tools

The course uses the Python ecosystem for practical ML work, including:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter / Google Colab
- Machine learning libraries and deployment tooling introduced throughout the course

---

## Progress Tracker

- [ ] Homework 01 — Introduction to Machine Learning
- [ ] Homework 02 — Regression
- [ ] Homework 03 — Classification
- [ ] Homework 04 — Evaluation Metrics
- [ ] Homework 05 — Deployment
- [ ] Homework 06 — Trees & Ensemble Learning
- [ ] Homework 08 — Neural Networks & Deep Learning
- [ ] Homework 09 — Serverless Deep Learning
- [ ] Homework 10 — Kubernetes
- [ ] Midterm Project
- [ ] Capstone 1
- [ ] Capstone 2

---

## Learning in Public

The course encourages sharing progress publicly through social media.

Suggested content:

- What was learned
- Key takeaways
- Difficulties encountered
- Interesting discoveries
- The next learning goal

Hashtag:

```text
#mlzoomcamp
```

Course contributors:

- [Alexey Grigorev](https://www.linkedin.com/in/agrigorev/)
- [DataTalksClub](https://www.linkedin.com/company/datatalks-club/)

---

## Resources

- [Machine Learning Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/)
- [Machine Learning Zoomcamp GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- [Homework 01](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01)
- [Course Calendar](https://courses.datatalks.club/ml-zoomcamp-2026/calendar.ics)

---

## Notes

This README is intended as a **high-level course overview and progress tracker**. Detailed explanations, experiments, answers, and code should remain inside each homework/project directory.
