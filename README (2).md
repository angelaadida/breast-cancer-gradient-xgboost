# 🎗️ Breast Cancer Classification — Gradient Boosting & XGBoost

Binary classification project using **Gradient Boosting** and **XGBoost** to predict whether a breast tumor is malignant or benign — achieving **97% accuracy**.

---

## 🎯 Problem Statement

Given 30 cell nucleus features from digitized breast mass images, classify tumors as:
- **0** = Malignant (cancerous)
- **1** = Benign (non-cancerous)

- **Dataset**: Scikit-learn Breast Cancer Wisconsin Dataset
- **Samples**: 569 records, 30 features
- **Train/Test split**: 80/20
- **Class distribution**: 63% Benign, 37% Malignant

---

## 🔄 ML Pipeline

### 1. Exploratory Data Analysis (EDA)
- No missing values in dataset
- Correlation heatmap of all 30 features
- Feature distribution plots (6×6 grid)
- Skewness & kurtosis analysis
- Outlier detection using IQR boxplots

### 2. Data Preprocessing
- IQR outlier capping on all numerical features
- RobustScaler normalization (resistant to outliers)
- 80/20 train/test split with random_state=42

### 3. Model Training & Evaluation

#### 🌲 Gradient Boosting Classifier
```
Parameters:
  n_estimators  : 100
  learning_rate : 1.0
  max_depth     : 1
```

| Metric | Score |
|--------|-------|
| **Accuracy** | **0.97** |
| Precision (Malignant) | 0.98 |
| Recall (Malignant) | 0.95 |
| F1-Score (Malignant) | 0.96 |
| Cross-val Average | 0.97 |

Confusion Matrix:
```
[[41  2]
 [ 1 70]]
```

#### ⚡ XGBoost Classifier
```
Parameters:
  objective     : binary:logistic
  n_estimators  : 100
  max_depth     : 3
  learning_rate : 0.1
```

| Metric | Score |
|--------|-------|
| **Accuracy** | **0.96** |
| Precision (Malignant) | 0.95 |
| Recall (Malignant) | 0.93 |
| F1-Score (Malignant) | 0.94 |
| Cross-val Average | 0.97 |

Confusion Matrix:
```
[[40  3]
 [ 2 69]]
```

---

## 🏆 Model Comparison

| Model | Accuracy | CV Average |
|-------|----------|------------|
| **GradientBoostingClassifier** | **0.97** | **0.97** |
| XGBClassifier | 0.96 | 0.97 |

**Winner**: GradientBoostingClassifier with **97.4% accuracy**

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Evaluation | Confusion Matrix, Classification Report, Cross-Validation |
| Preprocessing | RobustScaler, IQR Outlier Capping |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Open `Gradient_XGBoost_classifier_week6_2.ipynb` in Jupyter Notebook and run all cells.

---

## 📁 Project Structure

```
breast-cancer-gradient-xgboost/
│
├── Gradient_XGBoost_classifier_week6_2.ipynb   # Main notebook
└── README.md
```

> Dataset is loaded directly from `sklearn.datasets.load_breast_cancer()` — no download needed.

---

## 👩‍💻 Author

**Angela** — Data Scientist | AI • ML • GenAI • RAG  
📍 Kuala Lumpur, Malaysia  
🔗 [GitHub](https://github.com/angelaadida)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
