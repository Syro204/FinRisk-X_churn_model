# 🛡️ FinRisk AI: Enterprise Churn Intelligence

**FinRisk AI** is a specialized FinTech dashboard designed to identify at-risk bank customers using Machine Learning. Built with a high-performance **XGBoost** engine, it provides real-time churn probability scores, explainable AI insights, and bulk data processing capabilities.



---

## 🚀 Key Features

* **👤 Individual Risk Assessment:** Real-time inference for single customer profiles with automated risk labeling and probability metrics.
* **💡 Explainable AI (XAI):** Provides rule-based logic to explain the underlying factors (e.g., low tenure, security flags) behind a specific risk score.
* **📑 Universal Batch Processor:** Seamlessly ingest **Kaggle** or bank-standard CSV/Excel files with an automated **Data Sanitization Layer**.
* **🌐 Dynamic Data Mapping:** Automatically translates categorical geographical data into numeric features to ensure seamless model compatibility.
* **🔮 Retention Strategy Simulator:** A "What-If" analysis tool to simulate how interventions (e.g., loyalty plans) impact churn probability.
* **🎨 Monobank Dark UI:** A premium, dark-themed interface built for 2026 standards using the **Poppins** typography.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Frontend** | Streamlit |
| **ML Model** | XGBoost (Extreme Gradient Boosting) |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Seaborn, Matplotlib |
| **Serialization** | Joblib |

---

## 📂 Project Structure

```bash
FinTech_Churn_Project/
├── app.py              # Main Streamlit Dashboard (UI & Logic)
├── churn_model.pkl     # Pre-trained XGBoost Model
├── requirements.txt    # Python Dependencies
├── data/               # Sample Datasets (Kaggle/Internal)
└── README.md           # Project Documentation