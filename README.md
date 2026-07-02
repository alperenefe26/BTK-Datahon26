# BTK-Datahon26
Datathon project predicting career success using Machine Learning. Incorporates Turkish NLP on unstructured feedback, Lasso dimensionality reduction, feature engineering, and an optimized XGBoost model for high-accuracy MSE predictions.

# Career Success Prediction Pipeline (Datathon Project)

## Project Overview
This repository contains an end-to-end machine learning pipeline developed for the [Datathon 2026](https://www.kaggle.com/competitions/datathon-2026/overview) competition hosted on Kaggle. The objective is to predict candidates' career success scores (0-100) based on their internship performances, technical/HR interview results, and unstructured mentor feedback.

Instead of relying solely on baseline models, this project applies a rigorous statistical approach, natural language processing (NLP), and advanced gradient boosting techniques to minimize the Mean Squared Error (MSE) and build a robust, generalizable model.

## Key Methodologies & Workflow
* **Data Preprocessing & Imputation:** Handled missing values using KNN Imputation to preserve the statistical distribution of the dataset.
* **Natural Language Processing (NLP):** Extracted hidden success signals from unstructured mentor notes using Turkish Stemming and TF-IDF vectorization (300 features).
* **Dimensionality Reduction (Lasso Regression):** Applied LassoCV to penalize and eliminate 141 noisy/irrelevant features, preventing overfitting and reducing model variance.
* **Feature Engineering:** Created synthetic business-logic variables (e.g., *Hard/Soft Skills Ratio*, *Academic Risk Score*) to help tree-based models capture deeper non-linear relationships.
* **Hyperparameter Optimization:** Utilized `RandomizedSearchCV` on an **XGBoost Regressor** to find the optimal architecture (depth, learning rate, subsample) and achieve the best Cross-Validation MSE score.

## Transparency Note
The analytical problem-solving strategy, feature engineering logic, and statistical modeling architecture (Lasso to XGBoost transition) in this project were entirely designed and directed by me. During the development process, LLM (Large Language Model) tools were utilized as a coding assistant to accelerate the scripting process and maintain high coding standards.
