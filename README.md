# 💳 Credit Card Fraud Detection using Logistic Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-LogisticRegression-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A complete machine learning project that detects **credit card fraud** using **Logistic Regression**, with under-sampling to handle class imbalance, full EDA, visualizations, and evaluation metrics.

---

## 📊 Project Overview

| Item | Details |
|------|---------|
| **Algorithm** | Logistic Regression |
| **Dataset** | [Credit Card Fraud (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Samples** | ~284,807 transactions (highly imbalanced) |
| **Target** | `Class` – 0 = Legitimate, 1 = Fraud |
| **Balancing** | Under-sampling (492 fraud + 492 legitimate) |
| **Train/Test** | 80% / 20% (stratified on balanced set) |
| **Typical Accuracy** | Train ~94%, Test ~94% |

### Features
- `Time` – Seconds elapsed between transactions  
- `V1`–`V28` – PCA-transformed features (anonymized)  
- `Amount` – Transaction amount  
- `Class` – Fraud label (0/1)

---

## 📥 Dataset Setup

The full CSV is **~150 MB** and is **not** included in this repo (GitHub file size limits).

1. Download from Kaggle:  
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in the project root folder.

```bash
# Example
cd credit-card-fraud-detection
# put creditcard.csv here
```

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Add creditcard.csv (see Dataset Setup)

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

jupyter notebook Project_10_Credit_Card_Fraud_Detection.ipynb

# Full script (all figures + metrics)
python credit_card_fraud_detection.py
```

---

## 📈 What This Project Includes (A–Z)

### Data & Preprocessing
- Load full credit card dataset
- Class distribution (imbalanced)
- Statistical summary by class
- **Under-sampling** to create balanced dataset (equal fraud / legitimate)

### Visualizations
- Original class imbalance bar chart
- Balanced class distribution
- Amount distribution by class
- Confusion matrix heatmap

### Model & Metrics
- Logistic Regression (`max_iter=1000`)
- Accuracy (train & test)
- Classification report (precision, recall, F1)
- Confusion matrix

### Pipeline Notes
- Stratified train/test split on the balanced set
- Suitable baseline for highly imbalanced fraud detection

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── Project_10_Credit_Card_Fraud_Detection.ipynb
├── credit_card_fraud_detection.py
├── creditcard.csv              # You add this (from Kaggle)
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠️ Technologies

- Python 3.8+
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, metrics)

---

## 📝 License

MIT License – free to use and modify.

---

**Made with ❤️ for learning Machine Learning**
