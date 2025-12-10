import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import joblib

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------
df = pd.read_csv("/Users/rizwanashaik/Desktop/churn-analysis/telco_churn.csv")

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.fillna(df.median(numeric_only=True))

# ----------------------------------
# 2. ENCODING
# ----------------------------------
cat_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ----------------------------------
# 3. SPLIT
# ----------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 4. SCALING
# ----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------
# 5. MODELS
# ----------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# ----------------------------------
# 6. METRICS
# ----------------------------------
rf_report = classification_report(y_test, rf_pred, output_dict=True)
gb_report = classification_report(y_test, gb_pred, output_dict=True)

metrics = ["precision", "recall", "f1-score", "support"]

rf_values = [rf_report["weighted avg"][m] for m in metrics]
gb_values = [gb_report["weighted avg"][m] for m in metrics]

# ----------------------------------
# 7. VISUAL CHARTS (Saved as files)
# ----------------------------------

# ----- Chart 1: Accuracy -----
rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)

plt.figure(figsize=(6,4))
plt.bar(["Random Forest", "Gradient Boosting"], [rf_acc, gb_acc])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.close()

# ----- Chart 2: Precision -----
plt.figure(figsize=(6,4))
plt.bar(["Random Forest", "Gradient Boosting"], [rf_values[0], gb_values[0]])
plt.title("Precision Comparison (Weighted Avg)")
plt.ylabel("Precision")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("precision_comparison.png")
plt.close()

# ----- Chart 3: Recall -----
plt.figure(figsize=(6,4))
plt.bar(["Random Forest", "Gradient Boosting"], [rf_values[1], gb_values[1]])
plt.title("Recall Comparison (Weighted Avg)")
plt.ylabel("Recall")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("recall_comparison.png")
plt.close()

# ----- Chart 4: F1 Score -----
plt.figure(figsize=(6,4))
plt.bar(["Random Forest", "Gradient Boosting"], [rf_values[2], gb_values[2]])
plt.title("F1-Score Comparison (Weighted Avg)")
plt.ylabel("F1-score")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("f1score_comparison.png")
plt.close()

print("\nCharts saved successfully!")

# ----------------------------------
# 8. SAVE MODELS
# ----------------------------------
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(gb, "gradient_boosting_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Models saved successfully!")

# ----------------------------------
# 9. RECOMMEND BEST MODEL
# ----------------------------------
if rf_acc > gb_acc:
    best = "Random Forest"
else:
    best = "Gradient Boosting"

print(f"\n🔥 BEST MODEL: {best} (based on accuracy)")
from PIL import Image

# ----------------------------------
# 10. MERGE 4 PNG FILES INTO ONE IMAGE
# ----------------------------------

# Load images
img1 = Image.open("accuracy_comparison.png")
img2 = Image.open("precision_comparison.png")
img3 = Image.open("recall_comparison.png")
img4 = Image.open("f1score_comparison.png")

# Resize all images to the same width
width = 800
height = 500
img1 = img1.resize((width, height))
img2 = img2.resize((width, height))
img3 = img3.resize((width, height))
img4 = img4.resize((width, height))

# Create a blank combined image (2 rows × 2 columns)
combined = Image.new("RGB", (width * 2, height * 2), "white")

# Paste charts into grid
combined.paste(img1, (0, 0))
combined.paste(img2, (width, 0))
combined.paste(img3, (0, height))
combined.paste(img4, (width, height))

# Save final dashboard
combined.save("model_comparison_dashboard.png")

print("\nCombined dashboard saved as: model_comparison_dashboard.png")
