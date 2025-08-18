from flask import Flask, request, jsonify
from flask_cors import CORS
from model import FlightDelayModel
import os

app = Flask(__name__)
CORS(app)

# Initialize model
flight_model = FlightDelayModel()

# Train model on startup if not already trained
if not flight_model.load_model():
    print("Training new model...")
    if os.path.exists('data/airline_delay.csv'):
        flight_model.train('data/airline_delay.csv')
    else:
        print("Warning: Dataset not found at data/airline_delay.csv")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict_delay():
    try:
        data = request.json
        
        # Extract flight data
        flight_data = {
            'month': int(data.get('month', 1)),
            'day_of_week': int(data.get('day_of_week', 1)),
            'dep_time': int(data.get('dep_time', 800)),
            'distance': float(data.get('distance', 500)),
            'carrier': str(data.get('carrier', 'AA')),
            'origin': str(data.get('origin', 'JFK')),
            'dest': str(data.get('dest', 'LAX'))
        }
        
        # Get prediction
        delay_probability = flight_model.predict(flight_data)
        is_delayed = delay_probability > 0.5
        
        return jsonify({
            'delay_probability': delay_probability,
            'is_delayed': is_delayed,
            'confidence': max(delay_probability, 1 - delay_probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        if os.path.exists('data/airline_delay.csv'):
            accuracy = flight_model.train('data/airline_delay.csv')
            return jsonify({
                'message': 'Model retrained successfully',
                'accuracy': accuracy
            })
        else:
            return jsonify({'error': 'Dataset not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)