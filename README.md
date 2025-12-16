# 🎓 Early Student Dropout Prediction for Online Courses

A machine learning–powered web application that predicts **early dropout risk for new students in online courses** using **LMS engagement data from the first 4 weeks**.  
The goal is to enable **early intervention** rather than post-hoc analysis.

---

## 🚀 Live Demo
🔗 **Deployed App:** https://YOUR-APP-LINK.onrender.com

---

## 📌 Problem Statement

Student dropout is a major challenge in online education.  
Most existing analyses identify dropout **after the course has ended**, which limits the possibility of timely intervention.

This project focuses on **early identification of at-risk students** using only the **first 4 weeks of LMS activity**, making the prediction actionable for educators and institutions.

---

## 📊 Dataset

- **Dataset:** Open University Learning Analytics Dataset (OULAD)
- **Type:** Online & distance learning courses
- **Source:** https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- **Data Used:**
  - Student demographics
  - Early LMS engagement (clicks, activity days, interactions)
  - Final course outcomes

📌 Only **early-stage data** is used to prevent time leakage.

---

## 🧠 Methodology

### 🔹 Feature Engineering
- Aggregated LMS activity from the **first 4 weeks**
- Demographic attributes encoded using one-hot encoding
- Missing values handled explicitly

### 🔹 Target Variable
- **Dropout = 1:** Withdrawn or Failed  
- **Dropout = 0:** Passed or Distinction

### 🔹 Models Evaluated
- Logistic Regression (baseline)
- Random Forest (ensemble comparison)
- **XGBoost (final model)** ⭐

### 🔹 Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 🏆 Final Model Performance

| Rank | Model | Key Strength |
|-----|------|-------------|
| 🥇 | **XGBoost** | Best F1-score & ROC-AUC |
| 🥈 | Logistic Regression | Highest recall |
| 🥉 | Random Forest | Highest precision |

📌 **XGBoost** provides the best balance between precision and recall, making it ideal for early-warning systems.

---

## 🌐 Web Application

The project is deployed as a **Flask web application** with the following sections:

### ✔ Prediction (Main Feature)
- User input form
- Dropout probability score
- Risk classification (Low / Medium / High)

### ✔ How It Works
- Problem overview
- Dataset used
- Model & prediction window

### ✔ Project Links
- GitHub notebook & source code
- LinkedIn profile

This demonstrates **end-to-end ML deployment**, from data processing to real-world usage.

---

## 🛠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **Flask**
- **HTML / CSS**
- **Render (Deployment)**

---

## ⚙️ Setup & Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd Predictive-Analytics

# Install dependencies
pip install -r flask_app/requirements.txt

# Run the Flask app
cd flask_app
python app.py
```

Then open your browser and visit:
```
http://127.0.0.1:5000/
```

---

## 📁 Project Structure

```
Predictive-Analytics/
│
├── flask_app/
│   ├── app.py
│   ├── requirements.txt
│   ├── static/
│   │   ├── style.css
│   │   ├── logo.png
│   │   └── favicon.ico
│   └── templates/
│       └── index.html
│
├── model/
│   ├── xgb_model.pkl
│   └── feature_columns.pkl
│
├── notebook/
│   └── Early_Prediction_of_Student_Dropout.ipynb
│
├── .gitignore
└── README.md
```

---

## 🎯 Key Takeaways

- Demonstrates **early-stage predictive analytics**
- Prevents **data leakage**
- Shows **ML + deployment skills**
- Designed for **real-world educational intervention**

---

## 📜 License

This project is licensed under the **MIT License**.
