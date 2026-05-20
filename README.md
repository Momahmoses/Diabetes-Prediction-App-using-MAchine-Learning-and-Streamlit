# Diabetes Risk Prediction App

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Streamlit web app predicting diabetes risk using a Random Forest model — users input personal, health, and lifestyle data to receive a probability score, risk category, and top contributing factors for early assessment and prevention.

---

## Overview

This app enables non-clinical users to assess their diabetes risk in minutes. It uses a trained Random Forest classifier on the Pima Indians Diabetes dataset and explains predictions using feature importance — making the model interpretable for end users.

---

## Features

| Feature | Description |
|---------|-------------|
| Risk Probability Score | Likelihood of diabetes as a percentage |
| Risk Category | Low / Moderate / High classification |
| Top Contributing Factors | Feature importance for the individual prediction |
| Interactive Input Form | Glucose, BMI, age, insulin, blood pressure sliders |
| Streamlit Deployment | Accessible from any browser, no installation needed |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | scikit-learn (Random Forest) |
| App | Streamlit |
| Data | Pima Indians Diabetes dataset |
| Visualisation | Matplotlib, Seaborn |

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/Diabetes-Prediction-App-using-MAchine-Learning-and-Streamlit.git
cd Diabetes-Prediction-App-using-MAchine-Learning-and-Streamlit
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
streamlit run diabetes.py
```

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
