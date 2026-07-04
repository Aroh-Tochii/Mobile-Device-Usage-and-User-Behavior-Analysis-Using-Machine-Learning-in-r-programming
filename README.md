# Mobile Device Usage & User Behavior Analysis

An end-to-end machine learning system built in R that classifies mobile device users into 5 behavioral categories based on their usage patterns — deployed as an interactive Shiny web application.

## 🚀 Live Demo

**[Try the app here → https://aroh-tochii.shinyapps.io/mobile-behavior-intelligence/](https://aroh-tochii.shinyapps.io/mobile-behavior-intelligence/)**

## Overview

This project analyzes 700 mobile device user records to classify users into behavior categories ranging from Light to Extreme usage. Three machine learning models were trained, evaluated, and compared — with Random Forest selected for deployment based on accuracy and interpretability.

## Behavior Classes

| Class | Label | Description |
|---|---|---|
| 1 | Light User | Minimal daily usage — calls and basic tasks |
| 2 | Moderate User | Occasional usage throughout the day |
| 3 | Heavy User | Regular and frequent phone usage |
| 4 | Very Heavy User | Intensive daily usage — phone central to activities |
| 5 | Extreme User | Maximum usage — significant phone dependency |

## Tech Stack

| Layer | Tool |
|---|---|
| Language | R |
| Machine Learning | randomForest, e1071 (SVM), nnet (Logistic Regression) |
| Model Evaluation | caret (cross-validation, confusion matrix) |
| Visualization | ggplot2 |
| Dashboard | Shiny, shinydashboard |
| Deployment | shinyapps.io |

## Model Performance

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 90.71% | Baseline — interpretable |
| SVM | 95.71% | High accuracy across all classes |
| **Random Forest** | **95.71%** | **Selected — best accuracy + feature insights** |

**Baseline (random classifier): 20.71%** — improvement of +69% to +75% across all models.

## Shiny Application — 4 Tabs

**Predict Behavior** — Enter user profile via sliders and dropdowns. Get instant prediction with color-coded result, behavior description, and feature importance chart.

**Data Overview** — KPI cards, class distribution, screen time vs app usage scatter plot, feature distributions.

**Model Performance** — Model comparison chart, confusion matrix heatmap, full metrics table.

**About** — Project documentation, class definitions, methodology, tech stack.

## Key Findings

- App Usage Time and Data Usage are the strongest predictors of behavior class
- Age, Gender, and Device Model have minimal predictive power
- The dataset is perfectly balanced (140 records per class) — no class imbalance treatment needed
- 5-fold cross-validation confirmed no overfitting despite high accuracy

## Running Locally

```r
# Install dependencies
install.packages(c('shiny', 'shinydashboard', 'ggplot2', 'dplyr',
                   'caret', 'randomForest', 'e1071', 'nnet', 'caTools', 'tidyr'))

# Run the app
shiny::runApp('mobile_shiny_app.R')
```

## Dataset

700 records, 10 features — sourced from Kaggle (Mobile Device Usage and User Behavior Dataset by Vala Khorasani).

Features: Device Model, Operating System, App Usage Time, Screen On Time, Battery Drain, Number of Apps Installed, Data Usage, Age, Gender.

## Project Context

Originally developed as an academic machine learning project at the American University of Nigeria. Extended with a production-grade Shiny dashboard and deployed to shinyapps.io for public accessibility.

## Author

Tochukwu Aroh — Data Scientist & ML Engineer
