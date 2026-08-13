import sys
sys.modules['numexpr'] = None
sys.modules['bottleneck'] = None

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate():
    model_path = os.path.join(BASE_DIR, "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    data_path = os.path.join(BASE_DIR, "heart.csv")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[ERROR] File model.pkl atau scaler.pkl tidak ditemukan di {BASE_DIR}.")
        return

    if not os.path.exists(data_path):
        print(f"[ERROR] File dataset '{data_path}' tidak ditemukan.")
        return

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))


    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("EVALUASI PERFORMA MODEL SVM (HEART DISEASE)")
    print("=" * 60)
    print(f"--> Total Data Uji (Test Set) : {len(y_test)} sampel pasien")
    print(f"--> Akurasi Pengujian (Accuracy): {acc:.2%}")
    print("\nConfusion Matrix:")
    print(f"   [ TN: {cm[0][0]} | FP: {cm[0][1]} ]")
    print(f"   [ FN: {cm[1][0]} | TP: {cm[1][1]} ]")
    print("\nClassification Report Lengkap:")
    print(classification_report(y_test, y_pred, target_names=["Sehat/Normal (0)", "Risiko Jantung (1)"]))
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
