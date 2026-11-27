# Customer Loyalty Classification

[![tests](https://github.com/lad1xq/customer-loyalty-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/lad1xq/customer-loyalty-classification/actions/workflows/tests.yml)

Multiclass classification of retail customers into four loyalty segments:
`Occasional`, `Regular`, `Loyal`, and `High_Value`.

The project compares a manual one-vs-rest logistic regression trained with
batch gradient descent and Nesterov acceleration against scikit-learn's
`LogisticRegression` baseline.

## Results

The models were evaluated on the same stratified test set of 10,000 customers.

| Model | Accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|
| Manual gradient descent | 64.25% | 63.83% | 56.58% |
| Manual Nesterov | 65.80% | 64.68% | 60.70% |
| scikit-learn LogisticRegression | 66.43% | 64.90% | 62.98% |

Nesterov acceleration improved the manual implementation by 1.55 percentage
points in accuracy and by 4.12 points in macro recall, while remaining close
to the scikit-learn baseline.

## What is implemented

- train/test split with class stratification;
- preprocessing fitted only on the training data;
- median/mode imputation and one-hot encoding;
- feature standardization;
- manual one-vs-rest logistic regression;
- batch gradient descent and Nesterov acceleration;
- multiclass metrics and a scikit-learn baseline;
- automated tests for the model and preprocessing pipeline.

This is a supervised classification project: the segment labels already exist
in the dataset. It predicts those labels rather than discovering new clusters.

## Repository structure

```text
.
├── data/                 # Dataset download instructions
├── docs/                 # Project origin and contribution notes
├── notebooks/            # Exploratory notebook
├── scripts/              # Reproducible experiment entry point
├── src/                  # Reusable preprocessing and model code
└── tests/                # Automated tests
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset as described in [`data/README.md`](data/README.md), then run:

```bash
python scripts/run_experiment.py \
  --data data/retail_customer_segmentation.csv \
  --epochs 200
```

Run the tests:

```bash
python -m pytest -v
```

## Dataset

The project uses the [Online Retail Customer Segmentation Dataset](https://www.kaggle.com/datasets/rohit8527kmr7518/online-retail-customer-classification-dataset),
containing 50,000 customers and 14 demographic, behavioural, and transactional
columns. The dataset itself is not redistributed in this repository. Check the
license and download it from the original source.

## Project origin and my contribution

This work originated as a four-person university project at Innopolis
University. My contribution was the multiclass part: the manual one-vs-rest
model, gradient descent, Nesterov acceleration, the scikit-learn comparison,
and the experiment on training-set size.

The repository contains only the part I personally developed and a cleaned,
reproducible implementation based on it. See [`docs/project-origin.md`](docs/project-origin.md)
for the complete division of responsibilities.

