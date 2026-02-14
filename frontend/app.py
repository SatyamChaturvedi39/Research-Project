"""
NBA Trade Analyzer - Flask Backend with ML Integration
Loads the v2 XGBoost model and serves predictions
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pickle
import json
import numpy as np
import pandas as pd

# Load environment variables
load_dotenv()

# ============================================================================
# App Configuration
# ============================================================================
basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir)  # Go up to Research-Project folder

app = Flask(__name__,
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))

CORS(app)  # Enable CORS for API calls

# Environment variables
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'nba_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'players')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))


# ============================================================================
# Load ML Model and Artifacts
# ============================================================================
print("Loading ML model and artifacts...")

try:
    # Load multi-output model
    model_path = os.path.join(project_root, 'models', 'player_multioutput_v2.pkl')
    with open(model_path, 'rb') as f:
        ml_model = pickle.load(f)
    print("✓ Loaded multi-output model")
    
    # Load SHAP explainer
    explainer_path = os.path.join(project_root, 'models', 'shap_explainer_v2.pkl')
    with open(explainer_path, 'rb') as f:
        shap_explainer = pickle.load(f)
    print("✓ Loaded SHAP explainer")
    
    # Load feature names
    features_path = os.path.join(project_root, 'models', 'feature_names_v2.txt')
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(feature_names)} feature names")
    
    # Load target names
    targets_path = os.path.join(project_root, 'models', 'target_names_v2.txt')
    with open(targets_path, 'r') as f:
        target_names = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(target_names)} target names")
    
    # Load model metadata
    metadata_path = os.path.join(project_root, 'models', 'model_metadata_v2.json')
    with open(metadata_path, 'r') as f:
        model_metadata = json.load(f)
    print(f"✓ Loaded model metadata (version {model_metadata['model_version']})")
    
    # Load player data
    data_path = os.path.join(project_root, 'data', 'processed', 'player_features_v2_temporal.csv')
    players_df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(players_df)} player-season records")
    
    model_loaded = True
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("⚠️  App will run in demo mode without predictions")
    ml_model = None
    shap_explainer = None
    feature_names = []
    target_names = []
    model_metadata = {}
    players_df = None
    model_loaded = False

# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'model_version': model_metadata.get('model_version', 'N/A'),
        'features': len(feature_names),
        'targets': len(target_names)
    })

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of all teams"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Get unique teams from latest season, sorted alphabetically
    latest_season = players_df[players_df['season'] == '2024-25']
    teams = sorted(latest_season['team'].unique().tolist())
    
    return jsonify({
        'teams': teams,
        'count': len(teams)
    })

@app.route('/api/players', methods=['GET'])
def get_players():
    """Get list of available players, optionally filtered by team"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    team = request.args.get('team')
    
    # Filter by latest season
    latest_season = players_df[players_df['season'] == '2024-25']
    
    # Filter by team if provided
    if team:
        latest_season = latest_season[latest_season['team'] == team]
    
    # Sort by PPG descending
    players_list = latest_season.sort_values('points_per_game', ascending=False)[
        ['player_name', 'team', 'age', 'points_per_game']
    ].to_dict('records')
    
    return jsonify({
        'players': players_list,
        'team': team,
        'count': len(players_list)
    })

@app.route('/api/search', methods=['GET'])
def search_players():
    """Search for players by name"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    query = request.args.get('q', '').lower()
    
    if not query:
        return jsonify({'results': [], 'query': query})
    
    # Search in latest season
    latest_season = players_df[players_df['season'] == '2024-25']
    results = latest_season[
        latest_season['player_name'].str.lower().str.contains(query)
    ][['player_name', 'team', 'age', 'points_per_game']].head(10).to_dict('records')
    
    return jsonify({
        'results': results,
        'query': query
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Generate player performance predictions"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    player_name = data.get('player_name')
    
    if not player_name:
        return jsonify({'error': 'player_name required'}), 400
    
    # Find player in dataset
    player_data = players_df[
        players_df['player_name'].str.lower() == player_name.lower()
    ].sort_values('season', ascending=False)
    
    if len(player_data) == 0:
        return jsonify({'error': f'Player "{player_name}" not found'}), 404
    
    # Get most recent season data
    latest = player_data.iloc[0]
    
    # Calculate current age (accounting for time since last season)
    # Latest data is from 2024-25 season, calculate years passed
    from datetime import datetime
    current_year = datetime.now().year
    latest_season_year = int(latest['season'].split('-')[0]) + 2000  # "24-25" -> 2024
    years_passed = current_year - latest_season_year
    current_age = int(latest['age']) + years_passed
    
    # Check if we have all required features
    missing_features = [f for f in feature_names if f not in latest.index or pd.isna(latest[f])]
    
    if missing_features:
        return jsonify({
            'error': f'Insufficient data for {player_name}',
            'missing_features': missing_features[:5]  # Show first 5
        }), 400
    
    try:
        # Build feature vector
        feature_vector = np.array([[latest[f] for f in feature_names]])
        
        # Make prediction
        predictions = ml_model.predict(feature_vector)[0]
        
        # Create prediction dict
        pred_dict = {
            target_names[i].replace('target_next_', ''): round(float(predictions[i]), 2)
            for i in range(len(target_names))
        }
        
        # Get SHAP explanations for top factors
        shap_values = shap_explainer.shap_values(feature_vector)
        shap_dict = {
            name: round(float(val), 3)
            for name, val in zip(feature_names, shap_values[0])
        }
        
        # Get top 5 factors
        top_factors = dict(
            sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        )
        
        # Calculate confidence ranges using MAE from metadata
        confidence_ranges = {}
        for i, target in enumerate(target_names):
            target_key = target.replace('target_next_', '')
            mae = model_metadata['performance'].get(target, {}).get('test_mae', 2.0)
            
            pred_val = pred_dict[target_key]
            confidence_ranges[target_key] = [
                round(max(0, pred_val - mae), 2),  # Lower bound (non-negative)
                round(pred_val + mae, 2)  # Upper bound
            ]
        
        # Build response
        response = {
            'player_name': latest['player_name'],
            'team': latest['team'],
            'current_season': latest['season'],
            'last_season_year': f"20{latest['season'].split('-')[0]}-{latest['season'].split('-')[1]}",  # "24-25" -> "2024-2025"
            'current_age': current_age,  # Current age as of today
            'current_stats': {
                'ppg': round(float(latest['points_per_game']), 1),
                'rpg': round(float(latest['rebounds_per_game']), 1),
                'apg': round(float(latest['assists_per_game']), 1),
                'age': current_age  # Use calculated current age
            },
            'predictions': pred_dict,
            'confidence_ranges': confidence_ranges,
            'top_factors': top_factors,
            'model_version': model_metadata['model_version']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_version': model_metadata.get('model_version'),
        'training_date': model_metadata.get('training_date'),
        'features': len(feature_names),
        'targets': len(target_names),
        'performance': model_metadata.get('performance', {}),
        'sample_features': feature_names[:10],
        'target_list': target_names
    })

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("NBA TRADE ANALYZER - Flask Backend with ML")
    print("=" * 80)
    print(f"Model loaded: {model_loaded}")
    print(f"Features: {len(feature_names)}")
    print(f"Targets: {len(target_names)}")
    print("=" * 80)
    print("\nStarting server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    app.run(debug=True, port=5000)
