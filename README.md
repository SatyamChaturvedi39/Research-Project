<div align="center">
  <h1>🏀 NBA Trade Analyzer: An Explainable AI Trade Machine</h1>
  <p><i>A modern, data-driven methodology for evaluating professional basketball trades using Machine Learning, Monte Carlo simulations, and Explainable AI.</i></p>
</div>

---

## 🎯 What is this project?
If you've ever used a public NBA trade machine, you've probably realized its fatal flaw: **it only checks if a trade is financially legal.** You could theoretically trade LeBron James for five bench players, and the machine would stamp it "Successful." 

But in the real world, front offices need to know: *Is this trade actually good for my team? How will these incoming players perform when their usage rates drop? Will adding a third point guard ruin our offensive flow?*

The **NBA Trade Analyzer** solves this. Instead of a basic salary checker, this 3-tier web application uses a **Multi-Output XGBoost Machine Learning Model** trained on 5 years of historical box scores to mathematically project how a player's stats (PPG, RPG, APG, MPG, TS%) will shift. It calculates injury risks using a Random Forest classifier, scores the roster "fit," and runs a 500-instance **Monte Carlo Simulation** to estimate the net change in actual team wins. 

Finally, to avoid the dreaded "Black Box" problem of ML models, it uses **SHAP (SHapley Additive exPlanations)** to generate plain-English sentences explaining *why* the model made its prediction.

---

## 🚀 The Core Features

- **Multi-Stat Forecasting:** Predicts five distinct statistical avenues (Points, Rebounds, Assists, Minutes, True Shooting) simultaneously for up to 10 traded players.
- **Monte Carlo Win Projections:** Runs 500 probabilistic simulations factoring in standard deviation and injury risk to project exactly how many 82-game regular-season wins a team is gaining or losing.
- **Explainable AI (XAI):** Converts dense algorithmic math into readable insights (e.g., *"This player's age penalty is dragging his expected points down by 2.4 PPG"*).
- **Positional & Fit Penalties:** The engine literally gets mad at you if you try to exploit the math. Trade for too many Centers? You get hit with a "Positional Logjam Penalty." Trade for 3 players who all shoot 20 times a game? You trigger the "Scoring Cannibalization" penalty.
- **Cinematic UI/UX:** Built on React and Framer Motion, utilizing dark-mode "Glassmorphism" to look like a premium, proprietary front-office tool.

---

## 🧠 The Project Flow
Here is exactly what happens under the hood when you hit the **"Analyze Trade"** button:

1. **The Request:** The React frontend captures the selected players and teams, dropping them into a JSON payload sent to the Flask Rest API.
2. **Feature Extraction:** The Python backend queries MongoDB, locating the 54 historically engineered features (lag years, statistical trends, peak regression) for those specific players.
3. **Model Inference:** The vectors hit the serialized XGBoost regression model and the Random Forest medical classifier, predicting physical output and injury probabilities in milliseconds.
4. **SHAP Interpretation:** The TreeExplainer computes the combinatorial logic of the decision tree, ranking exactly which variables most heavily influenced the prediction.
5. **The Team Simulator:** The newly acquired players are algorithmically fused with their designated new team. The `trade_analyzer.py` engine checks for positional overlap, calculates roster depth health, and loops 500 simulations to find the median projected win delta.
6. **The Final Score:** A master formula compiles these vectors into a final 0-100 "Trade Score", shipping the package back to the UI to be visually rendered in SVG rings and dynamic accordions.

---

## 📊 The Dataset
Machine learning models are only as good as their data. We couldn't just feed the model "last year's points" and expect it to predict the future. 

Our dataset consists of over **1,600 individual player-seasons** extracted from the 2020 through 2025 NBA campaigns, transformed into **54 distinct features** per player:
- **Base Metrics:** The standard box score stats (PPG, RPG, MPG) plus advanced analytics like Usage Rate and True Shooting %.
- **Lag Features:** This was the game-changer. We created 5-year rolling windows (e.g., `ppg_lag1`, `ppg_lag5`). This forces the AI to acknowledge a player's long-term consistency floor.
- **Trend Slopes:** We calculated mathematical trajectories over 2, 3, and 4-year periods to teach the model the difference between a 22-year-old scoring 15 points (rising) versus a 34-year-old scoring 15 points (declining).
- **Career Anchors:** Metadata indicating how many years a player is removed from their absolute statistical peak.

*(Note: During holdout cross-validation testing simulating the entire 2024-25 season, the model proved exceptionally accurate, predicting Points Per Game with an $R^2$ of **0.919** and a Mean Absolute Error of **1.37** points).*

---

## 💻 Tech Stack

### 🔹 Frontend (The Face)
- **React 18** & **Vite** (Unbelievably fast HMR and compilation)
- **Tailwind CSS v3** (Utility-first styling powering the Glassmorphism aesthetic)
- **Framer Motion & GSAP** (Complex physics-based animations, loaders, and intro sequences)
- **Lucide React** (Clean, consistent iconography)

### 🔹 Backend (The Brains)
- **Python 3.12** & **Flask 3.0** (The connective tissue API router)
- **Scikit-Learn** & **XGBoost** (Core model mathematics and Random Forest classifications)
- **SHAP** (Game-theory based machine learning explainability)
- **Pandas** & **NumPy** (Lightning-fast dataset manipulation and Monte Carlo randomization)

### 🔹 Database (The Memory)
- **MongoDB** (NoSQL document ecosystem allowing hierarchical storage of yearly temporal stats and jagged injury logs without horrific SQL join constraints).

---

## ⚙️ How to Run Locally

If you want to spin up your own front-office terminal, you'll need two separate terminal windows.

### 1. Boot up the Backend Machine Learning Engine
Make sure you have your virtual environment activated and `requirements.txt` installed.
```bash
# Navigate to the root directory
cd Research-Project
# Start the Flask API & load the models into active memory
python app.py
```
*The server will initialize on `http://127.0.0.1:5000`*

### 2. Boot up the Front Desk (React UI)
```bash
# Open a second terminal and navigate to the modern frontend
cd frontend-modern
# Install npm packages if it's your first time
npm install
# Start the Vite development server
npm run dev
```
*The UI will launch on `http://localhost:5174`. Open this in your browser to start trading!*

---
*Created by Satyam Chaturvedi, Sahil, and Baarathi as part of a comprehensive Computer Science Software Engineering capstone project.*