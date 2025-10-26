# obesity_analysis_and_model.py
# Chạy: python obesity_analysis_and_model.py
# Hoặc copy vào 1 Jupyter / Colab cell và chạy cell.

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import pickle
import warnings
warnings.filterwarnings("ignore")

# --- 0. Đường dẫn file (chỉnh nếu cần) ---
DATA_PATH = "obesity/ObesityDataSet.csv"  # đường dẫn đến file dữ liệu
OUT_DIR = "obesity/output"  # thư mục để lưu kết quả
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. Load data ---
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# --- 2. EDA cơ bản ---
print("\n--- Missing values ---")
print(df.isnull().sum())
print("\n--- Dtypes ---")
print(df.dtypes)

# Phân phối lớp target: tìm cột target (tên phổ biến)
possible_targets = [c for c in df.columns if c.lower() in [
    'nobesity','obesity','obesity_level','obesitylevel',
    'obesity_levels','target','class','status','NObesity','Weight_Status'
]]
if possible_targets:
    target_col = possible_targets[0]
else:
    target_col = df.columns[-1]  # fallback
print("\nTarget column chosen:", target_col)
print(df[target_col].value_counts())

# Thống kê số liệu numeric
print("\n--- Describe numeric ---")
print(df.describe().T)

# Nếu cần vẽ hist/numeric correlations
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    corr = df[num_cols].corr()
    print("\nTop correlations (abs > 0.5):")
    corr_pairs = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
    print(corr_pairs[corr_pairs > 0.5].head(20))

# --- 3. Preprocessing ---
# 3.1: xử lý missing: ở đây dùng fillna trung bình/most_frequent tùy loại
for c in df.columns:
    if df[c].isnull().sum() > 0:
        if df[c].dtype in [np.float64, np.int64]:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])

# 3.2: categorical encoding (LabelEncoder cho các biến object)
cat_cols = df.select_dtypes(include=['object','bool']).columns.tolist()
if target_col in cat_cols:
    cat_cols.remove(target_col)
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    le_dict[c] = le

# 3.3: target encoding
target_le = LabelEncoder()
df[target_col] = target_le.fit_transform(df[target_col].astype(str))
print("\nMapped target classes:", dict(enumerate(target_le.classes_)))

# 3.4: tính BMI nếu có Height & Weight (tùy dataset)
if {'Weight','Height'}.issubset(df.columns) and 'BMI' not in df.columns:
    # giả sử Height cm
    df['BMI_computed'] = df['Weight'] / ((df['Height']/100).replace(0, np.nan)**2)
    df['BMI_computed'] = df['BMI_computed'].fillna(df['BMI_computed'].median())

# 3.5: loại bỏ các cột không cần thiết (ID,...)
drop_candidates = [c for c in df.columns if c.lower() in ['id','index','idx']]
df = df.drop(columns=drop_candidates, errors='ignore')

# 3.6: phân chia X,y
X = df.drop(columns=[target_col])
y = df[target_col].values

# 3.7: Scale numeric features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# --- 4. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# --- 5. Baseline models ---
models = {
    'LogisticRegression': LogisticRegression(multi_class='multinomial', max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVC': SVC(probability=True, random_state=42)
}

fitted = {}
for name, m in models.items():
    print(f"\nTraining {name} ...")
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=target_le.classes_))
    fitted[name] = m

# --- 6. Hyperparameter tuning (RandomForest example) ---
print("\n--- Hyperparam tuning for RandomForest (grid search) ---")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
gs = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
best_rf = gs.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Tuned RF accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=target_le.classes_, zero_division=0))

# --- 7. Confusion matrix & feature importance ---
cm = confusion_matrix(y_test, y_pred_rf)
print("Confusion matrix (rows=true, cols=pred):\n", cm)
plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix - Tuned RandomForest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(target_le.classes_)), target_le.classes_, rotation=45, ha='right')
plt.yticks(range(len(target_le.classes_)), target_le.classes_)
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, int(v), ha='center', va='center')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# Feature importances
if hasattr(best_rf, "feature_importances_"):
    fi = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features:\n", fi.head(20))
    fi.head(20).to_csv(os.path.join(OUT_DIR, "feature_importances_top20.csv"))

# --- 8. Cross-validated evaluation (final) ---
from sklearn.model_selection import cross_val_predict
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(best_rf, X, y, cv=cv)
print("\nCross-validated classification report:")
print(classification_report(y, y_cv_pred, target_names=target_le.classes_, zero_division=0))

# --- 9. Save final model & predictions ---
final_artifact = {
    "model": best_rf,
    "scaler": scaler,
    "label_encoders": le_dict,
    "target_le": target_le,
    "features": X.columns.tolist()
}
with open(os.path.join(OUT_DIR, "final_model.pkl"), "wb") as f:
    pickle.dump(final_artifact, f)

test_pred_df = X_test.copy()
test_pred_df[target_col + "_true"] = target_le.inverse_transform(y_test)
test_pred_df[target_col + "_pred"] = target_le.inverse_transform(y_pred_rf)
test_pred_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

print("\nSaved artifacts to:", OUT_DIR)
