<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/XGBoost-ML_Engine-FF6600?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-NoSQL-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
</p>

<h1 align="center">🏀 NBA Trade Analyzer</h1>
<h3 align="center">AI-Powered Trade Evaluation & Player Performance Forecasting</h3>

<p align="center">
  <i>Transforming raw basketball data into clear, actionable front-office intelligence.</i>
</p>

---

## 📌 About The Project

The **NBA Trade Analyzer** is a full-stack, machine learning-driven web application that quantitatively evaluates multi-player NBA trades. Instead of relying on subjective scouting and raw box-score averages, this system predicts **post-trade player performance**, assesses **injury risk**, simulates **team win projections**, and validates **salary cap feasibility** — all in under 3 seconds.

The application features **Explainable AI (SHAP)** to provide plain-English justifications for every prediction, ensuring complete transparency and trust for non-technical users.

> Built as a capstone research project at CHRIST (Deemed to be University), Department of Computer Science.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Multi-Output Stat Prediction** | Forecasts 5 key metrics — PPG, RPG, APG, MPG, TS% — using a 54-feature XGBoost model trained on 5 years of temporal NBA data (2020–2025). |
| **Explainable AI (XAI)** | SHAP-powered natural-language explanations tell users *why* a prediction was made (e.g., "His age of 34 is pulling the estimate down"). |
| **Injury Risk Classification** | A Random Forest classifier evaluates historical medical data to assign Low / Moderate / High / Very High injury risk categories. |
| **Monte Carlo Win Simulation** | Runs 500 probabilistic simulations per trade to project the net change in team wins (+/−) factoring in roster fit and injury risk. |
| **Salary Cap Validation** | Analyzes trade financial feasibility under NBA CBA salary matching rules. |
| **Positional Fit Penalties** | Detects scoring cannibalization and positional logjams to prevent artificially inflated projections. |
| **Cinematic UI** | Dark-mode Glassmorphism interface with GSAP/Framer Motion animations, gate-opening intro sequence, and fully mobile-responsive design. |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION TIER                        │
│         React 18  ·  Vite  ·  Tailwind CSS  ·  GSAP        │
│    ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐  │
│    │TeamPanel │ │TradeBlock│ │ Results   │ │  SHAP      │  │
│    │  .jsx    │ │  .jsx    │ │ Dashboard │ │ Explainer  │  │
│    └──────────┘ └──────────┘ └───────────┘ └────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │  REST API (JSON)
┌──────────────────────────▼──────────────────────────────────┐
│                   APPLICATION LOGIC TIER                     │
│            Flask 3.0  ·  Python 3.12  ·  SHAP               │
│    ┌──────────────┐ ┌─────────────┐ ┌───────────────────┐   │
│    │ XGBoost      │ │ Random      │ │  Monte Carlo      │   │
│    │ MultiOutput  │ │ Forest Inj. │ │  Win Simulator    │   │
│    │ Regressor    │ │ Classifier  │ │  (500 iterations) │   │
│    └──────────────┘ └─────────────┘ └───────────────────┘   │
│    ┌──────────────┐ ┌─────────────┐                         │
│    │ Trade Score  │ │ Salary Cap  │                         │
│    │ Engine       │ │ Validator   │                         │
│    └──────────────┘ └─────────────┘                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                       DATA TIER                              │
│              MongoDB  ·  CSV Feature Caches                  │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│    │ players  │ │  teams   │ │ injuries │ │ salary_data  │  │
│    └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

### Feature Engineering (54 Features)

The model does not simply use last season's stats. It processes **54 carefully engineered temporal features** organized into four categories:

| Category | Count | Examples |
| :--- | :---: | :--- |
| **Base Stats** | 16 | `age`, `points_per_game`, `true_shooting_pct`, `usage_rate`, `points_per_minute` |
| **Lag Features** | 26 | `ppg_lag1` → `ppg_lag5`, `mpg_lag1` → `mpg_lag5`, `ts_pct_lag1` → `ts_pct_lag3` |
| **Trend Features** | 4 | `ppg_trend_2yr`, `ppg_trend_3yr`, `ppg_trend_4yr`, `mpg_trend_2yr` |
| **Career Features** | 8 | `peak_ppg`, `years_since_peak_ppg`, `career_ppg_avg`, `ppg_coefficient_variation` |

### Model Accuracy (Backtested Against 2024–25 Season)

| Target Metric | MAE | R² Score | Interpretation |
| :--- | :---: | :---: | :--- |
| Points Per Game | 1.37 | 0.919 | Within ~1 basket of actual scoring |
| Rebounds Per Game | 0.52 | 0.908 | Highly reliable |
| Assists Per Game | 0.36 | 0.926 | Excellent precision |
| Minutes Per Game | 2.81 | 0.847 | Strong correlation |
| True Shooting % | 0.04 | -1.21 | High variance — known limitation |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/api/health` | System health check and model status |
| `GET` | `/api/teams` | List all 30 NBA franchises |
| `GET` | `/api/players?team=LAL` | Fetch active roster for a team |
| `POST` | `/api/predict` | Predict next-season stats for a player |
| `POST` | `/api/trade/evaluate` | Full trade evaluation with Monte Carlo sim |
| `GET` | `/api/model/info` | Model metadata, feature list, and accuracy |

---

## 🛠️ Tech Stack

### Backend
- **Language:** Python 3.12
- **Framework:** Flask 3.0
- **ML Engine:** XGBoost, Scikit-learn
- **Explainability:** SHAP (TreeExplainer)
- **Data Processing:** Pandas, NumPy
- **Database:** MongoDB (pymongo)

### Frontend
- **Framework:** React 18
- **Build Tool:** Vite
- **Styling:** Tailwind CSS v3
- **Animations:** Framer Motion, GSAP
- **Charts:** Recharts
- **Icons:** Lucide React

---

## 📁 Project Structure

```
NBA-Trade-Analyzer/
├── app.py                      # Flask REST API (main backend entry point)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (MongoDB URI, ports)
│
├── models/                     # Serialized ML model binaries
│   ├── player_multioutput_v2.pkl
│   ├── injury_clf.pkl
│   ├── shap_explainers_v2.pkl
│   └── model_metadata_v2.json
│
├── services/
│   └── trade_analyzer.py       # Core trade logic (fit penalties, Monte Carlo)
│
├── data/
│   ├── processed/              # Engineered temporal CSV feature sets
│   └── team_overrides_2025_26.json
│
├── notebooks/                  # Jupyter training notebooks
│   ├── M2.ipynb                # Team stats integration & model training
│   ├── M3_Injury_Model.ipynb   # Injury risk classifier
│   └── M3_Salary_Cap.ipynb     # Salary cap analysis
│
├── frontend-modern/            # React SPA
│   ├── src/
│   │   ├── App.jsx             # Main application state & routing
│   │   ├── api.js              # Axios API client
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── Hero.jsx
│   │   │   ├── IntroAnimation.jsx
│   │   │   ├── TeamPanel.jsx
│   │   │   ├── TradeBlock.jsx
│   │   │   ├── ResultsDashboard.jsx
│   │   │   ├── ModelInfo.jsx
│   │   │   └── About.jsx
│   │   └── index.css           # Global Glassmorphism design system
│   └── tailwind.config.js
│
└── scripts/                    # Data population & utility scripts
```

---

## 👥 Team

| Member | Module |
| :--- | :--- |
| **Satyam Chaturvedi** | Core ML Engine (XGBoost), Flask API, React Frontend, SHAP Integration |
| **Sahil** | Team Statistics Analysis & Model Integration (M2) |
| **Baarathi** | Injury Risk Classification & Salary Cap Analysis (M3) |

---

## 📄 License

This project is developed for academic and research purposes at CHRIST (Deemed to be University), Department of Computer Science.