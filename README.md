# 🧪 LiftLab — Production Causal Inference Engine

> **Distinguish genuine causal lift from correlation, seasonal effects, and selection bias — at production scale.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![EconML](https://img.shields.io/badge/EconML-0.15-orange.svg)](https://econml.azurewebsites.net)
[![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)](htmlcov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The $2M Problem

E-commerce growth teams run 50+ promotions per quarter. Standard A/B tools return p-values — but when randomization is impossible, observational data is confounded, or populations shift between experiments, those p-values are measuring **correlation, not causation**.

The result: teams scale promotions that show positive correlation but **zero true causal lift**, misallocating millions in promotion spend.

LiftLab solves this with a production-grade causal inference engine that combines rigorous identification strategies, heterogeneous treatment effect estimation, and automated drift detection.

---

## Live Demo

| Service                    | URL                                               | Description                      |
| -------------------------- | ------------------------------------------------- | -------------------------------- |
| 🎨 **Streamlit Dashboard** | [localhost:8501](http://localhost:8501)           | Growth team experiment hub       |
| 🚀 **FastAPI Docs**        | [localhost:8080/docs](http://localhost:8080/docs) | Interactive API documentation    |
| 📊 **MLflow UI**           | [localhost:5001](http://localhost:5001)           | Experiment tracking & versioning |

---

## How It Works
