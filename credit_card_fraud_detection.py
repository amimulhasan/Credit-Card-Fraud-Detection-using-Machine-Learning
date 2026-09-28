"""
Credit Card Fraud Detection using Logistic Regression
======================================================
Full pipeline: load → EDA → under-sample → train → evaluate → plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def load_dataset():
    for path in [
        "creditcard.csv",
        "credit_data.csv",
        "/content/creditcard.csv",
        "/content/credit_data.csv",
    ]:
        if os.path.exists(path):
            print(f"Loading: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Dataset not found.\n"
        "Download creditcard.csv from:\n"
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "and place it in the project folder."
    )


def main():
    print("=" * 60)
    print("  Credit Card Fraud Detection | Logistic Regression")
    print("=" * 60)

    # 1. Load
    print("\n[1/8] Loading dataset...")
    credit_card_data = load_dataset()
    print(f"Shape: {credit_card_data.shape}")
    print("\nFirst 5 rows:\n", credit_card_data.head())
    print("\nMissing values:", credit_card_data.isnull().sum().sum())
    print("\nClass distribution:\n", credit_card_data["Class"].value_counts())

    # 2. Imbalance plot
    print("\n[2/8] Class imbalance plot...")
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Class", data=credit_card_data, palette=["#2ecc71", "#e74c3c"])
    plt.title("Original Class Distribution (Imbalanced)")
    plt.xlabel("Class (0=Legitimate, 1=Fraud)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_imbalance.png", dpi=150)
    print("Saved: class_imbalance.png")
    plt.close()

    # 3. Separate classes & under-sample
    print("\n[3/8] Under-sampling for balanced dataset...")
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    print(f"Legitimate: {legit.shape[0]} | Fraud: {fraud.shape[0]}")

    n_fraud = fraud.shape[0]
    legit_sample = legit.sample(n=n_fraud, random_state=2)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    new_dataset = new_dataset.sample(frac=1, random_state=2).reset_index(drop=True)

    print(f"Balanced dataset shape: {new_dataset.shape}")
    print(new_dataset["Class"].value_counts())
    print("\nMean by class:\n", new_dataset.groupby("Class").mean())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Class", data=new_dataset, palette=["#2ecc71", "#e74c3c"])
    plt.title("Balanced Class Distribution (After Under-sampling)")
    plt.xlabel("Class (0=Legitimate, 1=Fraud)")
    plt.tight_layout()
    plt.savefig("balanced_distribution.png", dpi=150)
    print("Saved: balanced_distribution.png")
    plt.close()

    # Amount by class
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Class", y="Amount", data=new_dataset, palette=["#2ecc71", "#e74c3c"])
    plt.title("Transaction Amount by Class (Balanced Set)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("amount_by_class.png", dpi=150)
    print("Saved: amount_by_class.png")
    plt.close()

    # 4. Split
    print("\n[4/8] Train / Test split (80/20, stratified)...")
    X = new_dataset.drop(columns="Class", axis=1)
    Y = new_dataset["Class"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 5. Train
    print("\n[5/8] Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=2)
    model.fit(X_train, Y_train)
    print("Model trained successfully!")

    # 6. Evaluate
    print("\n[6/8] Evaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)

    print(f"\nAccuracy on Training data : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Accuracy on Test data     : {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\nClassification Report (Test Set):")
    print(
        classification_report(
            Y_test,
            test_pred,
            target_names=["Legitimate (0)", "Fraud (1)"],
        )
    )

    # 7. Confusion matrix
    print("\n[7/8] Confusion matrix...")
    cm = confusion_matrix(Y_test, test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")
    plt.close()

    # 8. Sample prediction
    print("\n[8/8] Sample prediction...")
    sample = X_test.iloc[[0]]
    actual = Y_test.iloc[0]
    pred = model.predict(sample)[0]
    print(f"Actual Class    : {actual} ({'Fraud' if actual == 1 else 'Legitimate'})")
    print(f"Predicted Class : {pred} ({'Fraud' if pred == 1 else 'Legitimate'})")

    print("\n" + "=" * 60)
    print("Done! Generated plots:")
    print("  - class_imbalance.png")
    print("  - balanced_distribution.png")
    print("  - amount_by_class.png")
    print("  - confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
