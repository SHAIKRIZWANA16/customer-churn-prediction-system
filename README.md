📊 Telco Customer Churn Prediction  
A Machine Learning project that predicts whether a telecom customer will churn using classification models.  
This project compares two powerful models — **Random Forest** and **Gradient Boosting** — and generates a complete visual analysis dashboard.

---

## 🚀 Project Overview  
Customer churn is a major challenge for telecom companies. This project analyzes Telco customer data to predict churn and identify key patterns affecting customer retention.

The goal is to:
- Build ML models to predict churn  
- Compare model performance using multiple evaluation metrics  
- Generate visual charts  
- Produce a combined dashboard summarizing model results  
- Recommend the best-performing model  

---

## 📁 Project Structure  
```

├── churn.py
├── telco_churn.csv
├── random_forest_model.pkl
├── gradient_boosting_model.pkl
├── scaler.pkl
├── accuracy_comparison.png
├── precision_comparison.png
├── recall_comparison.png
├── f1score_comparison.png
├── model_comparison_dashboard.png
└── README.md

```

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib  
- PIL (Pillow)  
- Joblib  

---

## 📌 Features  
✔️ Loads and preprocesses Telco churn dataset  
✔️ Encodes categorical features  
✔️ Scales numeric variables  
✔️ Trains two ML models  
   - **Random Forest Classifier**  
   - **Gradient Boosting Classifier**  
✔️ Evaluates models using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Support  

✔️ Generates 4 comparison charts  
✔️ Automatically merges charts into **one dashboard PNG**  
✔️ Saves trained models as `.pkl` files  
✔️ Prints the **best model** recommendation  

---

## 🧠 Model Comparison  
The project compares model performance based on the accuracy score:

- If **Random Forest > Gradient Boosting** → Random Forest is recommended  
- Otherwise → Gradient Boosting is chosen  

Final dashboard preview (example):

```

+----------------+----------------+
|   Accuracy     |   Precision    |
+----------------+----------------+
|     Recall     |    F1 Score    |
+----------------+----------------+

````

---

## ▶️ Run the Project  
### 1. Install dependencies  
```bash
pip install -r requirements.txt
````

### 2. Run the script

```bash
python3 churn.py
```

### 3. Output files generated

* accuracy_comparison.png
* precision_comparison.png
* recall_comparison.png
* f1score_comparison.png
* model_comparison_dashboard.png
* random_forest_model.pkl
* gradient_boosting_model.pkl
* scaler.pkl

---

## 📝 Dataset

This project uses the **Telco Customer Churn dataset** containing customer demographics, service details, account information, and churn labels.

---

## 📸 Screenshots

You may include:

* Model comparison dashboard
* Accuracy charts
* Sample predictions

---

## 🏆 Best Model Result

The script prints the best-performing model at the end, using accuracy as the main metric.

---
