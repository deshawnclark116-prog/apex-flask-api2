from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load models (upload .pkl files next step)
models = {}
model_files = ['nba_model.pkl', 'nfl_model.pkl', 'mlb_model.pkl', 'nhl_model.pkl']

for model_file in model_files:
    if os.path.exists(model_file):
        sport = model_file.replace('_model.pkl', '').upper()
        models[sport] = pickle.load(open(model_file, 'rb'))

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "models": list(models.keys())})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sport = data['sport']
    games = data['games']
    
    if sport not in models:
        return jsonify({"error": f"No {sport} model"}), 400
    
    model = models[sport]
    predictions = []
    
    for game in games:
        features = np.zeros(20)  # TODO: your real features
        win_prob = model.predict_proba([features])[0][1]
        
        predictions.append({
            'modelPrediction': {
                'homeWinProb': float(win_prob * 100),
                'spread': -4.2,
                'total': 227.8,
                'confidenceLow': float((win_prob - 0.12) * 100),
                'confidenceHigh': float((win_prob + 0.12) * 100),
                'homeFavorite': win_prob > 0.5
            },
            'topEVBets': [],
            'playerProps': []
        })
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
