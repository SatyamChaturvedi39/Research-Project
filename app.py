"""
NBA Trade Analyzer - Production Flask Backend
Serves ML predictions and frontend interface
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# ============================================================================
# App Configuration
# ============================================================================
app = Flask(__name__,
            template_folder='frontend/templates',
            static_folder='frontend/static')

CORS(app)  # Enable CORS for API calls

# Environment variables
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'nba_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'players')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))

# ============================================================================
# Database Connection & Data Loading
# ============================================================================

def load_data_from_db():
    """
    Load player data from MongoDB and flatten it into a DataFrame
    Compatible with the ML model's expected format.
    """
    print(f"Connecting to MongoDB at {DATABASE_NAME}...")
    try:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI not set")
            
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Check if empty
        if collection.count_documents({}) == 0:
            print("⚠ MongoDB is empty. Falling back to CSV.")
            return pd.read_csv('data/processed/player_features_v2_temporal.csv')
            
        print("Fetching documents from MongoDB...")
        cursor = collection.find({})
        
        flattened_rows = []
        for doc in cursor:
            player_name = doc.get('player_name', 'Unknown')
            # The 'stats' array contains the historical rows (features)
            stats = doc.get('stats', [])
            for s in stats:
                s['player_name'] = player_name
                flattened_rows.append(s)
        
        df = pd.DataFrame(flattened_rows)
        print(f"✓ Successfully loaded {len(df)} rows from MongoDB")
        client.close()
        return df
        
    except Exception as e:
        print(f"⚠ Database connection failed: {e}")
        print("Falling back to local CSV file...")
        try:
            return pd.read_csv('data/processed/player_features_v2_temporal.csv')
        except:
            print("❌ Critical: Could not load data from DB or CSV")
            return None

# ============================================================================
# Load ML Model and Artifacts
# ============================================================================
print("Loading ML model and artifacts...")

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
    
    # Load model metadata
    with open('models/model_metadata_v2.json', 'r') as f:
        model_metadata = json.load(f)
    print(f"✓ Loaded model metadata (version {model_metadata.get('model_version')})")
    
    # LOAD DATA FROM MONGODB INSTEAD OF CSV
    players_df = load_data_from_db()
    
    if players_df is not None:
        latest_season = players_df['season'].max()
        print(f"✓ Loaded {len(players_df)} player-season records")
        print(f"✓ Latest season detected: {latest_season}")
        model_loaded = True
    else:
        model_loaded = False
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("⚠️  App will run in demo mode without predictions")
    ml_model = None
    shap_explainer = None
    feature_names = []
    target_names = []
    model_metadata = {}
    players_df = None
    latest_season = None
    model_loaded = False

# ============================================================================
# Helper Functions
# ============================================================================

def get_latest_season_data():
    """Get data from the most recent season dynamically"""
    if not model_loaded or players_df is None:
        return None
    return players_df[players_df['season'] == latest_season]

def calculate_current_age(dataset_age, dataset_season):
    """
    Calculate player's current age accounting for time since season
    
    Args:
        dataset_age: Age from dataset (e.g., 39)
        dataset_season: Season string (e.g., '24-25')
    
    Returns:
        Current age as of today
    """
    current_year = datetime.now().year
    
    # Parse season year (e.g., '24-25' -> 2024, '2024-25' -> 2024)
    season_parts = dataset_season.split('-')
    season_start_year_str = season_parts[0]
    season_start_year = int(season_start_year_str)
    
    # Handle 2-digit years if necessary
    if len(season_start_year_str) == 2:
        if season_start_year < 50:
            season_start_year += 2000
        else:
            season_start_year += 1900
            
    years_passed = current_year - season_start_year
    return int(dataset_age) + years_passed

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
        'status': 'ok' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'model_version': model_metadata.get('model_version', 'N/A'),
        'features': len(feature_names),
        'targets': len(target_names),
        'latest_season': latest_season,
        'player_records': len(players_df) if players_df is not None else 0
    })

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of all teams from latest season"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    season_data = get_latest_season_data()
    if season_data is None or len(season_data) == 0:
        return jsonify({'error': 'No data available'}), 500
    
    teams = sorted(season_data['team'].unique().tolist())
    
    # NBA Team Mapping (Abbreviation -> Full Name)
    team_map = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets', 'BRK': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets', 'CHO': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
        'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns', 'PHO': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }
    
    formatted_teams = [{'code': t, 'name': team_map.get(t, t)} for t in teams]
    
    return jsonify({
        'teams': formatted_teams,
        'count': len(teams),
        'season': latest_season
    })

@app.route('/api/players', methods=['GET'])
def get_players():
    """Get list of available players, optionally filtered by team"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    team = request.args.get('team')
    
    # Get latest season data
    season_data = get_latest_season_data()
    if season_data is None or len(season_data) == 0:
        return jsonify({'error': 'No data available'}), 500
    
    # Filter by team if provided
    if team:
        season_data = season_data[season_data['team'] == team]
    
    # Sort by PPG descending
    players_list = season_data.sort_values('points_per_game', ascending=False)[
        ['player_name', 'team', 'age', 'points_per_game']
    ].to_dict('records')
    
    return jsonify({
        'players': players_list,
        'team': team,
        'count': len(players_list),
        'season': latest_season
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
    season_data = get_latest_season_data()
    if season_data is None:
        return jsonify({'results': [], 'query': query})
    
    # Case-insensitive search
    matches = season_data[
        season_data['player_name'].str.lower().str.contains(query, na=False)
    ][['player_name', 'team', 'points_per_game']].to_dict('records')
    
    return jsonify({
        'results': matches[:10],  # Limit to 10 results
        'query': query
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict player performance for next season
    Uses latest available data and ML model
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    player_name = data.get('player_name', '').strip()
    
    if not player_name:
        return jsonify({'error': 'Player name required'}), 400
    
    # Find player (case-insensitive)
    player_data = players_df[
        players_df['player_name'].str.lower() == player_name.lower()
    ].sort_values('season', ascending=False)
    
    if len(player_data) == 0:
        return jsonify({'error': f'Player "{player_name}" not found'}), 404
    
    # Get most recent season data
    latest = player_data.iloc[0]
    
    # Calculate current age
    current_age = calculate_current_age(latest['age'], latest['season'])
    
    # Parse season for display (e.g., '24-25' -> '2024-2025')
    season_parts = latest['season'].split('-')
    display_season_start = int(season_parts[0])
    if display_season_start < 50:
        display_season_start += 2000
    else:
        display_season_start += 1900
    display_season = f"{display_season_start}-{display_season_start + 1}"
    
    # Check if we have all required features
    missing_features = [f for f in feature_names if f not in latest.index or pd.isna(latest[f])]
    
    if missing_features:
        return jsonify({
            'error': f'Insufficient data for {player_name}',
            'missing_features': missing_features[:5]
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
            'last_season_year': display_season,
            'current_age': current_age,
            'current_stats': {
                'ppg': round(float(latest['points_per_game']), 1),
                'rpg': round(float(latest['rebounds_per_game']), 1),
                'apg': round(float(latest['assists_per_game']), 1),
                'age': current_age
            },
            'predictions': pred_dict,
            'confidence_ranges': confidence_ranges,
            'top_factors': top_factors,
            'model_version': model_metadata['model_version']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model metadata and performance metrics"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'metadata': model_metadata,
        'feature_count': len(feature_names),
        'target_count': len(target_names),
        'latest_season': latest_season,
        'sample_features': feature_names[:10],
        'target_names': target_names
    })

# ============================================================================
# Startup Banner
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("NBA TRADE ANALYZER - Flask Backend with ML")
    print("="*80)
    print(f"Model loaded: {model_loaded}")
    print(f"Features: {len(feature_names)}")
    print(f"Targets: {len(target_names)}")
    if latest_season:
        print(f"Latest Season: {latest_season}")
        print(f"Player Records: {len(players_df)}")
    print("="*80)
    print(f"\nStarting server at http://127.0.0.1:{FLASK_PORT}")
    print("Press Ctrl+C to stop")
    print("="*80 + "\n")
    
    app.run(host='127.0.0.1', port=FLASK_PORT, debug=True)
