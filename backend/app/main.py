"""
FastAPI Backend for NBA Trade Recommendation System
Updated for v2 Multi-Output Model with 54 Temporal Features
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pickle
import json
import numpy as np
from datetime import datetime

# Import database connection
from .database import players_collection, predictions_collection, ping_db

# ============================================================================
# Initialize FastAPI App
# ============================================================================
app = FastAPI(
    title="NBA Player Performance Prediction API",
    description="Predicts multiple player performance metrics using 5-season temporal features",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Load Model and Artifacts
# ============================================================================
print("Loading model v2 and artifacts...")

try:
    # Load multi-output model
    with open('models/player_multioutput_v2.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    print("✓ Loaded multi-output model")
    
    # Load SHAP explainer
    with open('models/shap_explainer_v2.pkl', 'rb') as f:
        shap_explainer = pickle.load(f)
    print("✓ Loaded SHAP explainer")
    
    # Load feature names
    with open('models/feature_names_v2.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(feature_names)} feature names")
    
    # Load target names
    with open('models/target_names_v2.txt', 'r') as f:
        target_names = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(target_names)} target names")
    
    # Load metadata
    with open('models/model_metadata_v2.json', 'r') as f:
        model_metadata = json.load(f)
    print(f"✓ Loaded model metadata (version {model_metadata['model_version']})")
    
    model_loaded = True
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("⚠️  API will run in limited mode without predictions")
    ml_model = None
    shap_explainer = None
    feature_names = []
    target_names = []
    model_metadata = {}
    model_loaded = False

# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    database: bool
    model_loaded: bool
    model_version: Optional[str]
    features_count: int
    targets_count: int

class PredictionRequest(BaseModel):
    player_name: str

class PredictionResponse(BaseModel):
    player_name: str
    team: str
    current_season: str
    predictions: Dict[str, float]
    confidence_ranges: Dict[str, List[float]]
    top_factors: Dict[str, float]
    model_version: str

# ============================================================================
# Helper Functions
# ============================================================================

def build_feature_vector(player_history: List[Dict]) -> np.ndarray:
    """
    Build 54-feature vector from player's historical data
    
    Args:
        player_history: List of season stats, ordered from oldest to newest
        
    Returns:
        numpy array of shape (1, 54)
    """
    
    if len(player_history) == 0:
        raise ValueError("No player history provided")
    
    # Latest season is the "current" season
    current = player_history[-1]
    
    # Initialize feature dict
    features = {}
    
    # ========== BASE FEATURES (16) ==========
    features['age'] = current.get('age', 0)
    features['games_played'] = current.get('games_played', 0)
    features['minutes_per_game'] = current.get('minutes_per_game', 0)
    features['points_per_game'] = current.get('points_per_game', 0)
    features['rebounds_per_game'] = current.get('rebounds_per_game', 0)
    features['assists_per_game'] = current.get('assists_per_game', 0)
    features['steals_per_game'] = current.get('steals_per_game', 0)
    features['blocks_per_game'] = current.get('blocks_per_game', 0)
    features['turnovers_per_game'] = current.get('turnovers_per_game', 0)
    features['field_goal_pct'] = current.get('field_goal_pct', 0)
    features['free_throw_pct'] = current.get('free_throw_pct', 0)
    features['true_shooting_pct'] = current.get('true_shooting_pct', 0)
    features['points_per_minute'] = current.get('points_per_minute', 0)
    features['usage_rate'] = current.get('usage_rate', 0)
    features['assist_rate'] = current.get('assist_rate', 0)
    features['rebound_rate'] = current.get('rebound_rate', 0)
    
    # ========== LAG FEATURES (29) ==========
    # PPG lags (5)
    for lag in [1, 2, 3, 4, 5]:
        idx = len(player_history) - 1 - lag
        features[f'ppg_lag{lag}'] = player_history[idx]['points_per_game'] if idx >= 0 else 0
    
    # MPG lags (5)
    for lag in [1, 2, 3, 4, 5]:
        idx = len(player_history) - 1 - lag
        features[f'mpg_lag{lag}'] = player_history[idx]['minutes_per_game'] if idx >= 0 else 0
    
    # RPG lags (3)
    for lag in [1, 2, 3]:
        idx = len(player_history) - 1 - lag
        features[f'rpg_lag{lag}'] = player_history[idx]['rebounds_per_game'] if idx >= 0 else 0
    
    # APG lags (3)
    for lag in [1, 2, 3]:
        idx = len(player_history) - 1 - lag
        features[f'apg_lag{lag}'] = player_history[idx]['assists_per_game'] if idx >= 0 else 0
    
    # Other lags
    idx = len(player_history) - 2
    features['spg_lag1'] = player_history[idx]['steals_per_game'] if idx >= 0 else 0
    features['bpg_lag1'] = player_history[idx]['blocks_per_game'] if idx >= 0 else 0
    
    for lag in [1, 2, 3]:
        idx = len(player_history) - 1 - lag
        features[f'games_lag{lag}'] = player_history[idx]['games_played'] if idx >= 0 else 0
        
    for lag in [1, 2, 3]:
        idx = len(player_history) - 1 - lag
        features[f'ts_pct_lag{lag}'] = player_history[idx]['true_shooting_pct'] if idx >= 0 else 0
    
    for lag in [1, 2]:
        idx = len(player_history) - 1 - lag
        features[f'fg_pct_lag{lag}'] = player_history[idx]['field_goal_pct'] if idx >= 0 else 0
    
    # ========== TREND FEATURES (9) ==========
    # 2-year trend
    if len(player_history) >= 3:
        features['ppg_trend_2yr'] = (player_history[-1]['points_per_game'] - 
                                       player_history[-3]['points_per_game']) / 2
        features['mpg_trend_2yr'] = (player_history[-1]['minutes_per_game'] - 
                                       player_history[-3]['minutes_per_game']) / 2
    else:
        features['ppg_trend_2yr'] = 0
        features['mpg_trend_2yr'] = 0
    
    # 3-year trend
    if len(player_history) >= 4:
        features['ppg_trend_3yr'] = (player_history[-1]['points_per_game'] - 
                                       player_history[-4]['points_per_game']) / 3
    else:
        features['ppg_trend_3yr'] = 0
    
    # 4-year trend
    if len(player_history) >= 5:
        features['ppg_trend_4yr'] = (player_history[-1]['points_per_game'] - 
                                       player_history[-5]['points_per_game']) / 4
    else:
        features['ppg_trend_4yr'] = 0
    
    # Career stats
    all_ppg = [s['points_per_game'] for s in player_history]
    all_games = [s['games_played'] for s in player_history]
    all_mpg = [s['minutes_per_game'] for s in player_history]
    
    features['seasons_in_dataset'] = len(player_history)
    features['career_ppg_avg'] = np.mean(all_ppg)
    features['career_ppg_std'] = np.std(all_ppg) if len(all_ppg) > 1 else 0
    features['career_games_avg'] = np.mean(all_games)
    features['career_mpg_avg'] = np.mean(all_mpg)
    
    # Peak and consistency
    peak_ppg = max(all_ppg)
    peak_idx = all_ppg.index(peak_ppg)
    features['peak_ppg'] = peak_ppg
    features['years_since_peak_ppg'] = len(player_history) - 1 - peak_idx
    features['ppg_coefficient_variation'] = (np.std(all_ppg) / np.mean(all_ppg)) if np.mean(all_ppg) > 0 else 0
    
    # Convert to numpy array in correct order
    feature_vector = np.array([[features[name] for name in feature_names]])
    
    return feature_vector

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "NBA Player Performance Prediction API",
        "version": "2.0.0",
        "model_version": model_metadata.get('model_version', 'N/A'),
        "endpoints": {
            "health": "/health",
            "players": "/players",
            "search": "/players/search?q={query}",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    db_ok = await ping_db()
    
    return HealthResponse(
        status="ok" if (db_ok and model_loaded) else "degraded",
        database=db_ok,
        model_loaded=model_loaded,
        model_version=model_metadata.get('model_version'),
        features_count=len(feature_names),
        targets_count=len(target_names)
    )

@app.get("/players")
async def get_players(limit: int = 50, skip: int = 0):
    """Get list of players"""
    players = []
    cursor = players_collection.find(
        {}, 
        {"_id": 0, "player_name": 1, "team": 1, "position": 1, "age": 1}
    ).skip(skip).limit(limit)
    
    async for doc in cursor:
        players.append(doc)
    
    return {"players": players, "count": len(players)}

@app.get("/players/search")
async def search_players(q: str, limit: int = 10):
    """Search for players by name"""
    cursor = players_collection.find(
        {"player_name": {"$regex": q, "$options": "i"}},
        {"_id": 0, "player_name": 1, "team": 1, "position": 1, "age": 1}
    ).limit(limit)
    
    results = []
    async for doc in cursor:
        results.append(doc)
    
    return {"results": results, "query": q}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict player performance for next season
    Requires player to have at least 3 seasons of historical data
    """
    
    if not model_loaded:
        raise HTTPException(503, "Model not loaded")
    
    # Find player in database
    player = await players_collection.find_one(
        {"player_name": {"$regex": f"^{request.player_name}$", "$options": "i"}}
    )
    
    if not player:
        raise HTTPException(404, f"Player '{request.player_name}' not found")
    
    # Get player's historical stats
    if "stats" not in player or len(player["stats"]) < 3:
        raise HTTPException(
            400, 
            f"Insufficient data for {request.player_name}. Need at least 3 seasons of history."
        )
    
    try:
        # Build feature vector from player history
        player_history = player["stats"]
        feature_vector = build_feature_vector(player_history)
        
        # Make prediction
        predictions = ml_model.predict(feature_vector)[0]
        
        # Create prediction dict
        pred_dict = {
            target_names[i].replace('target_next_', ''): round(float(predictions[i]), 2)
            for i in range(len(target_names))
        }
        
        # Get SHAP explanations
        shap_values = shap_explainer.shap_values(feature_vector)
        shap_dict = {
            name: round(float(val), 3) 
            for name, val in zip(feature_names, shap_values[0])
        }
        
        # Get top 5 factors
        top_factors = dict(
            sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        )
        
        # Calculate confidence ranges (using MAE from metadata)
        confidence_ranges = {}
        for i, target in enumerate(target_names):
            target_key = target.replace('target_next_', '')
            mae = model_metadata['performance'].get(target, {}).get('test_mae', 2.0)
            
            pred_val = pred_dict[target_key]
            confidence_ranges[target_key] = [
                round(pred_val - mae, 2),
                round(pred_val + mae, 2)
            ]
        
        # Save prediction to database
        pred_doc = {
            "player_name": player["player_name"],
            "team": player["team"],
            "predictions": pred_dict,
            "confidence_ranges": confidence_ranges,
            "top_factors": top_factors,
            "model_version": model_metadata['model_version'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await predictions_collection.insert_one(pred_doc)
        
        # Return response
        return PredictionResponse(
            player_name=player["player_name"],
            team=player["team"],
            current_season=player_history[-1].get("season", "2024-25"),
            predictions=pred_dict,
            confidence_ranges=confidence_ranges,
            top_factors=top_factors,
            model_version=model_metadata['model_version']
        )
        
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model metadata and performance metrics"""
    if not model_loaded:
        raise HTTPException(503, "Model not loaded")
    
    return {
        "metadata": model_metadata,
        "feature_names": feature_names,
        "target_names": target_names,
        "feature_count": len(feature_names),
        "target_count": len(target_names)
    }

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 80)
    print("NBA PLAYER PERFORMANCE PREDICTION API v2.0")
    print("=" * 80)
    print(f"Model loaded: {model_loaded}")
    print(f"Features: {len(feature_names)}")
    print(f"Targets: {len(target_names)}")
    print("=" * 80)
