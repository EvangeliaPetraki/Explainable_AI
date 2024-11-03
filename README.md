# Explainable AI for Rist Prediction

This repository contains two scripts implementing explainable AI (XAI) techniques for a risk prediction model. The goal is to train machine learning classifiers on health-related data and interpret their predictions using SHAP and LIME explainability frameworks. This is especially useful in high-stakes domains, such as healthcare, where understanding model predictions is crucial for trust and accountability.


## Project Overview

This project applies explainable AI techniques to a binary risk prediction model, trained on a health dataset with an imbalanced target variable (UnderRisk). Since many real-world datasets exhibit imbalance between classes, synthetic minority oversampling (SMOTE) is used to address this issue.

**SMOTE (Synthetic Minority Over-sampling Technique)** helps balance the dataset by generating synthetic examples for the minority class, increasing the model’s ability to learn patterns from less-represented data without introducing bias. By oversampling the minority class, SMOTE improves the model’s predictive accuracy and generalization on imbalanced data. This enables both the **CatBoost Classifier with SHAP** and **Random Forest Classifier with LIME** models to perform effectively and fairly.


## Overview of the Models

* **CatBoost Classifier with SHAP**: CatBoost is a gradient boosting algorithm optimized for categorical features. SHAP values help interpret model predictions by attributing each prediction to individual features, showing their positive or negative influence.
* **Random Forest Classifier with LIME and Instance Map**: A Random Forest model is trained and explained with LIME (Local Interpretable Model-Agnostic Explanations) and a feature importance instance map. This highlights which features contribute most to each individual prediction, allowing for in-depth inspection at the instance level.
Both models are trained on a health dataset to predict if an individual is under risk (UnderRisk), with a range of health-related features as input variables.


## Libraries Used
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn
* catboost
* shap
* lime
* imbalanced-learn (for handling imbalanced data using SMOTE)

## Scripts Overview

### 1. [CatBoost_SHAP.py](https://github.com/EvangeliaPetraki/Explainable_AI/blob/main/CatBoost_SHAP.py)
Purpose: Train a CatBoost classifier on the dataset and use SHAP values to explain predictions.

**Key Components:** 

* **Data Preprocessing**: The data is loaded, missing values are checked, and SMOTE is used to balance classes.
* **Model Training**: Three CatBoost models are trained with different numbers of iterations (5, 10, and 15) to optimize performance.
* **Explainability with SHAP**:
  * Global interpretability: SHAP bar plots highlight feature importance across the entire dataset.
  * Local interpretability: Individual prediction explanations via force plots and waterfall plots.

### 2. [RandomForest_LIME.py](https://github.com/EvangeliaPetraki/Explainable_AI/blob/main/RandomForest_LIME)
Purpose: Train a Random Forest classifier on the same dataset and apply LIME, along with an instance feature importance map, to explain individual predictions.

**Key Components:**

* **Data Preprocessing**: Balancing the dataset with SMOTE, checking feature distributions, and performing exploratory data analysis (EDA).
* **Model Training**: Three Random Forest models are trained with 5, 10, and 50 trees respectively.
* **Explainability with LIME and Feature Importance Instance Map**:
  * Feature importance plot: Visualize feature importances across all predictions.
  * Instance-level explanations: Generate explanations for two sample instances using LIME, highlighting influential features in each case.
  * Instance Map: Feature importance values from the Random Forest model are mapped for specific instances to help identify which features influence the prediction for each individual instance.
 
## Results

* **CatBoost + SHAP**:
  * SHAP Summary plots and bar plots illustrate which features are most influential in predicting UnderRisk.
  * Force and waterfall plots provide detailed explanations for individual predictions, highlighting the exact contribution of each feature.

* **Random Forest + LIME and Instance Map**:
  * The feature importance plot shows which features drive predictions across the dataset.
  * LIME plots for individual predictions highlight how specific features impact predictions for sample instances.
  * Instance-level importance maps provide a detailed view of the feature importance for each individual case, allowing a direct visual comparison of influential features in specific predictions.

## References: 
* [CatBoost Documentation](https://catboost.ai/)
* [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
* [LIME Documentation](https://lime.readthedocs.io/en/latest/)
* [SMOTE Technique](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
