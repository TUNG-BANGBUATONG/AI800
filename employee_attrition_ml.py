#!/usr/bin/env python3
"""
Employee Attrition Prediction (การทำนายการลาออกของพนักงาน)
เปรียบเทียบ 3 โมเดล Machine Learning:
1. Logistic Regression
2. Random Forest Classifier
3. Naive Bayes (GaussianNB)
(+ SVM Classifier เป็นโมเดลทางเลือกเพิ่มเติม)

ตัวชี้วัด: Accuracy, Precision, Recall, F1-Score และ Confusion Matrix
Dataset: IBM HR Analytics Employee Attrition & Performance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# -------------------------------------------------------------
# 1. โหลดข้อมูล (Load Dataset)
# -------------------------------------------------------------
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์ {DATA_PATH} กรุณาตรวจสอบว่ามีไฟล์อยู่ในไดเรกทอรีเดียวกัน")

df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("1. ข้อมูลภาพรวม (Overview):")
print(f"จำนวนแถว (ตัวอย่าง): {df.shape[0]}, จำนวนคอลัมน์ (คุณลักษณะ): {df.shape[1]}")
print(f"จำนวนค่าสูญหาย (Missing Values): {df.isnull().sum().sum()}")
print("\nการแจกแจงของตัวแปรเป้าหมาย (Target: Attrition):")
print(df["Attrition"].value_counts())
print(df["Attrition"].value_counts(normalize=True) * 100)

# -------------------------------------------------------------
# 2. การเตรียมข้อมูลและคัดเลือกคุณลักษณะ (Data Preprocessing)
# -------------------------------------------------------------
# ตัดคอลัมน์ที่มีค่าคงที่เดิม หรือเป็นเพียงรหัสระบุตัวตน (No predictive value)
drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
df_cleaned = df.drop(columns=[col for col in drop_cols if col in df.columns])

# แปลง Target Attrition: 'Yes' -> 1, 'No' -> 0
df_cleaned["Attrition"] = df_cleaned["Attrition"].map({"Yes": 1, "No": 0})

# แยกประเภทคอลัมน์เชิงกลุ่ม (Categorical) และแปลงเป็น One-Hot Encoding
cat_cols = df_cleaned.select_dtypes(include=["object", "string"]).columns.tolist()
df_encoded = pd.get_dummies(df_cleaned, columns=cat_cols, drop_first=True)

# แยก Feature Matrix (X) และ Target Vector (y)
X = df_encoded.drop(columns=["Attrition"])
y = df_encoded["Attrition"]

print("\n" + "=" * 60)
print("2. การเตรียมข้อมูล (Preprocessing Completed):")
print(f"ขนาดของ Feature Matrix (X): {X.shape}")
print(f"ขนาดของ Target (y): {y.shape}")

# -------------------------------------------------------------
# 3. การแบ่งชุดข้อมูลสอนและทดสอบ (Train-Test Split)
# -------------------------------------------------------------
# ใช้ test_size = 0.2 (80% Train, 20% Test)
# ใช้ stratify=y เพื่อรักษาอัตราส่วนคลาส 1 และ 0 ในทั้งสองชุดข้อมูล
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("3. ขนาดของชุดข้อมูลฝึกสอนและทดสอบ:")
print(f"Train set: {X_train.shape[0]} ตัวอย่าง ({y_train.sum()} ลาออก / {len(y_train) - y_train.sum()} ไม่ลาออก)")
print(f"Test set:  {X_test.shape[0]} ตัวอย่าง ({y_test.sum()} ลาออก / {len(y_test) - y_test.sum()} ไม่ลาออก)")

# ทำ Feature Scaling ด้วย StandardScaler สำหรับโมเดลที่ต้องการระยะห่าง (Distance-based / Gradient-based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------
# 4. กำหนดโมเดลและฝึกสอน (Model Training)
# -------------------------------------------------------------
# กำหนด 3 โมเดลหลักตามโจทย์ + SVM เป็นโมเดลเสริม
models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "use_scaled": True,
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_estimators=100),
        "use_scaled": False,  # Tree-based ไม่จำเป็นต้องทำ scaling
    },
    "Naive Bayes (Gaussian)": {
        "model": GaussianNB(),
        "use_scaled": True,
    },
    "SVM (RBF Kernel)": {
        "model": SVC(random_state=42, probability=True),
        "use_scaled": True,
    },
}

results = []
conf_matrices = {}
y_preds = {}
y_probs = {}

print("\n" + "=" * 60)
print("4. กำลังฝึกสอนและประเมินผลแต่ละโมเดล...")

for name, config in models.items():
    clf = config["model"]
    x_tr = X_train_scaled if config["use_scaled"] else X_train
    x_te = X_test_scaled if config["use_scaled"] else X_test

    clf.fit(x_tr, y_train)
    y_pred = clf.predict(x_te)
    y_preds[name] = y_pred

    # หาความน่าจะเป็นสำหรับ ROC Curve
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(x_te)[:, 1]
    else:
        y_prob = clf.decision_function(x_te)
    y_probs[name] = y_prob

    # คำนวณตัวชี้วัด
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices[name] = cm

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
    })

# จัดทำตารางเปรียบเทียบ
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("5. ตารางเปรียบเทียบประสิทธิภาพของโมเดล (Model Comparison Table):")
print("-" * 60)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("-" * 60)

# บันทึกตารางเป็น CSV
results_df.to_csv("model_comparison_metrics.csv", index=False)
print("บันทึกตารางผลการทดสอบลงไฟล์: model_comparison_metrics.csv")

# -------------------------------------------------------------
# 5. แสดงผล Confusion Matrix ของทุกโมเดล
# -------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Confusion Matrix Comparison - Employee Attrition Prediction", fontsize=16, fontweight="bold")

model_names = list(models.keys())
for idx, ax in enumerate(axes.flatten()):
    if idx < len(model_names):
        name = model_names[idx]
        cm = conf_matrices[name]
        
        # วาด Heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{name}\nAcc: {results_df.loc[results_df['Model'] == name, 'Accuracy'].values[0]:.2%} | F1: {results_df.loc[results_df['Model'] == name, 'F1-Score'].values[0]:.2%}", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.close()
print("บันทึกภาพ Confusion Matrix: confusion_matrices.png")

# -------------------------------------------------------------
# 6. กราฟเปรียบเทียบค่า Metrics แต่ละโมเดล (Bar Chart)
# -------------------------------------------------------------
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
df_plot = pd.melt(results_df, id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_plot, x="Metric", y="Score", hue="Model", palette="tab10")
plt.title("Performance Metrics Comparison Across Models", fontsize=14, fontweight="bold")
plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt="%.2f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=300)
plt.close()
print("บันทึกภาพกราฟแท่งเปรียบเทียบตัวชี้วัด: metrics_comparison.png")

# -------------------------------------------------------------
# 7. กราฟ ROC-AUC Curves
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, y_probs[name])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2)

plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300)
plt.close()
print("บันทึกภาพ ROC Curves: roc_curves.png")

# -------------------------------------------------------------
# 8. Feature Importance จาก Random Forest
# -------------------------------------------------------------
rf_model = models["Random Forest"]["model"]
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importance.head(15).values, y=feature_importance.head(15).index, palette="viridis", hue=feature_importance.head(15).index, legend=False)
plt.title("Top 15 Most Important Features (Random Forest)", fontsize=14, fontweight="bold")
plt.xlabel("Relative Importance Score", fontsize=12)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()
print("บันทึกภาพ Feature Importance: feature_importance.png")

print("\n" + "=" * 60)
print("รันครบถ้วนทุกขั้นตอนเรียบร้อย!")
