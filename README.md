# Production System README

## NBA Trade Analyzer - ML-Powered Player Performance Prediction

A production-ready Flask web application that predicts NBA player performance using a 54-feature XGBoost model with SHAP explanations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open browser
http://127.0.0.1:5000
```

## Features

- ✅ Multi-output ML model predicting 5 performance metrics
- ✅ 54 temporal features from 5 seasons of data
- ✅ SHAP explanations for interpretability
- ✅ Dynamic season detection (no hardcoded dates)
- ✅ Real-time trade analysis
- ✅ Professional UX with modal-based player selection

## Structure

```
/app.py              # Main Flask backend
/models/             # ML model artifacts
/data/               # Dataset storage
/notebooks/          # Jupyter notebooks for training
/frontend/           # HTML/CSS/JS assets
  /templates/        # Jinja2 templates
  /static/           # Static files
/.env                # Environment configuration
/requirements.txt    # Python dependencies
```

## API Endpoints

- `GET /` - Homepage
- `GET /api/health` - System health check
- `GET /api/teams` - List all NBA teams
- `GET /api/players?team=LAL` - Get players by team
- `POST /api/predict` - Predict player performance
- `GET /api/model/info` - Model metadata

## Tech Stack

- **Backend:** Flask 3.0, Python 3.12+
- **ML:** XGBoost, Scikit-learn, SHAP
- **Data:** Pandas, NumPy
- **Frontend:** Vanilla JS, HTML5, CSS3
- **Database:** MongoDB (optional)

## Model Performance

- **PPG Prediction:** R² = 0.826, MAE = 2.43
- **Multi-Output:** Predicts PPG, RPG, APG, MPG, TS%
- **Features:** 54 (16 base + 26 lag + 12 trend)
- **Training:** 600 samples, 150 test samples

## Production Features

- Dynamic latest season detection
- Smart age calculation accounting for time
- Startup data validation
- Comprehensive error handling
- Health check endpoints
- CORS enabled for API access

## Author

Research Project - Module 1: Player Performance Prediction

## License

For academic/research use