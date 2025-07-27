# Credit Risk and Loan Approval Analysis (Advanced Deep Learning)

This project uses an Artificial Neural Network (ANN) to predict both **Loan Approval** (classification) and **Credit Risk Score** (regression) using financial and personal data from applicants.

---

## 📌 Project Overview

- **Goal**: Predict if a loan should be approved and estimate the applicant's risk score.
- **Approach**: Multi-output deep learning model (classification + regression).
- **Dataset**: `financial-risk-for-loan-approval.csv` from Kaggle.

---

## 📁 Files in This Repository

- `Loan.csv`: Dataset used for training and testing.
- `FinancialLoanRisk_ML_Project.ipynb`: Full Jupyter Notebook with data processing, model building, and evaluation.
- `Details.docx`: Project explanation.

---

## 🔍 Key Steps Performed

1. **Data Preprocessing**
   - Dropped irrelevant features like `ApplicationDate`.
   - One-hot encoded categorical variables.
   - Scaled numeric features using `MinMaxScaler`.

2. **Model Architecture (Keras Functional API)**
   - Batch Normalization for stability.
   - Two hidden layers with ReLU activation.
   - Dual outputs:
     - `Loan_approved`: Binary classification (`sigmoid`)
     - `Risk_score`: Regression (`ReLU`)

3. **Training & Evaluation**
   - EarlyStopping used to avoid overfitting.
   - Trained for ~13 epochs with strong validation performance.
   - Evaluation metrics:
     - Accuracy: **94%**
     - Risk Score MAE: **~2.8**
     - Precision/Recall (Loan Approval): Class 0 = 99%, Class 1 = 79%

4. **Final Thoughts**
   - The model prioritizes reducing false approvals — crucial in financial risk.
   - Deep learning improved predictive power over traditional models.

---

## 📊 Results Snapshot

| Metric                  | Value    |
|------------------------|----------|
| Accuracy               | 94%      |
| Risk Score MAE         | ~2.8     |
| Class 0 Recall         | 99%      |
| Class 1 Recall         | 79%      |

---

## ✅ Tools & Libraries

- Python, Pandas, Matplotlib, Scikit-learn
- TensorFlow & Keras

---

## 👨‍💻 Author

**Karri Vamsi**  
Feel free to connect or suggest improvements!
