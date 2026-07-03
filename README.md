# 🚗 AI-Powered Driver Behaviour & Stall Prediction System

An end-to-end Machine Learning project that predicts vehicle stall risk from synthetic driving telemetry using **XGBoost**. The project combines **feature engineering**, **explainable AI (SHAP)**, and **driver behaviour profiling (K-Means)** to identify high-risk driving conditions and analyse driving patterns.

---

## ✨ Key Features

- Generated a **150,000-row synthetic vehicle telemetry dataset** across 50 drivers and 500 driving sessions.
- Engineered **14 mechanical and behavioural features**, including rolling statistics and interaction features.
- Trained an **XGBoost multiclass classifier** for stall risk prediction.
- Applied **SHAP Explainability** to identify the most influential features affecting model predictions.
- Performed **K-Means clustering** to group drivers into four behavioural profiles based on driving characteristics.
- Produced visual reports for model explainability and driver profiling.

---

## 📊 Results

| Metric | Value |
|--------|------:|
| Dataset Size | 150,000 samples |
| Drivers | 50 |
| Sessions | 500 |
| Engineered Features | 14 |
| Model | XGBoost |
| Classification Accuracy | 99% |
| 5-Fold Cross Validation | 98.6% |
| Driver Profiles | 4 |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Joblib

---

## 📁 Project Structure

```text
driver-behaviour-stall-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── REPORT.pdf
│
├── models/
│
├── reports/
│
├── src/
│   ├── data_gen.py
│   ├── features.py
│   ├── train.py
│   ├── shap_explains.py
│   └── driver_profiling.py
│
├── README.md
└── requirements.txt
```

---

## 📂 Repository Contents

- Synthetic telemetry data generation
- Feature engineering pipeline
- XGBoost model training
- SHAP explainability analysis
- Driver behaviour profiling using K-Means
- Technical project report

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/driver-behaviour-stall-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project modules:

```bash
python src/data_gen.py
python src/features.py
python src/train.py
python src/shap_explains.py
python src/driver_profiling.py
```

---

## 🔮 Future Improvements

- Real-time vehicle telemetry integration
- Embedded deployment on ESP32
- CAN Bus support
- Interactive Streamlit dashboard
- Live inference using onboard sensor data

---

## 👤 Author

**Skandha Pavithra S**

B.Tech Electronics & Communication Engineering  
Vellore Institute of Technology
