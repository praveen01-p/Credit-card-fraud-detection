# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using multiple classification algorithms and evaluating their performance with metrics such as F1-score and ROC-AUC.

## Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small fraction of the overall transactions.

This project explores different machine learning and neural-network-based approaches for identifying fraudulent transactions and compares their performance using classification and ranking metrics.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle.

The dataset contains anonymized credit card transactions with:

* 284,807 transactions
* 492 fraudulent transactions
* 30 input features
* `Class` as the target variable

The dataset is highly imbalanced, making accuracy alone insufficient for evaluating fraud detection performance.

## Methodology

### 1. Data Preprocessing

The data preprocessing pipeline includes:

* Separating features and target variable
* Standardizing the `Amount` feature using `StandardScaler`
* Handling the severe class imbalance using **Random Undersampling**
* Splitting the processed data into training and testing sets

### 2. Classification Models

The following models were implemented and compared:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)
* Neural Network using TensorFlow/Keras

### 3. Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Precision-Recall Curve

F1-score and ROC-AUC are particularly useful for this problem because of the significant class imbalance.

## Results

The implemented models achieved approximately the following results on the test data:

| Model               | Accuracy | F1-Score | ROC-AUC |
| ------------------- | -------: | -------: | ------: |
| Logistic Regression |     0.94 |     0.92 |    0.96 |
| SVM                 |     0.94 |     0.92 |    0.97 |
| Random Forest       |     0.95 |     0.93 |    0.97 |
| XGBoost             |     0.95 |     0.93 |    0.97 |
| MLP                 |     0.95 |     0.94 |    0.98 |
| Neural Network      |     0.95 |     0.94 |    0.98 |

The MLP and TensorFlow/Keras neural-network approaches achieved the highest reported F1-score and ROC-AUC among the evaluated models.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* TensorFlow
* Keras
* Matplotlib
* Seaborn
* Imbalanced-learn
* Jupyter Notebook

## Project Workflow

```text
Raw Transaction Data
        ↓
Data Preprocessing
        ↓
Feature Scaling
        ↓
Random Undersampling
        ↓
Train/Test Split
        ↓
Multiple Classification Models
        ↓
Predictions
        ↓
F1 / ROC-AUC / Precision / Recall
        ↓
Model Comparison
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost tensorflow keras matplotlib seaborn imbalanced-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Credit Card Fraud Detection Using Neural Networks.ipynb
```

and run the notebook cells sequentially.

## Key Learning Outcomes

* Understanding highly imbalanced classification problems
* Applying random undersampling to address class imbalance
* Comparing classical ML, ensemble, boosting, and neural-network models
* Evaluating fraud detection models using F1-score and ROC-AUC
* Interpreting confusion matrices and precision-recall curves
* Understanding the trade-off between precision and recall in fraud detection

## Future Improvements

* Compare additional imbalance-handling techniques such as SMOTE
* Perform systematic hyperparameter tuning
* Explore anomaly-detection approaches
* Evaluate models using cost-sensitive learning
* Build a real-time fraud detection API
* Deploy the best-performing model as a web service

## Author

**Boda Praveen**
B.Tech Electrical Engineering
Indian Institute of Technology Madras
