import React, { useState } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    month: '1',
    day_of_week: '1',
    dep_time: '800',
    distance: '500',
    carrier: 'AA',
    origin: 'JFK',
    dest: 'LAX'
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });
      
      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Error:', error);
      setPrediction({ error: 'Failed to get prediction' });
    }
    
    setLoading(false);
  };

  const carriers = ['AA', 'DL', 'UA', 'WN', 'AS', 'B6', 'NK', 'F9'];
  const airports = ['JFK', 'LAX', 'ORD', 'DFW', 'DEN', 'SFO', 'SEA', 'LAS', 'PHX', 'IAH'];

  return (
    <div className="app">
      <div className="container">
        <h1>Flight Delay Predictor</h1>
        <p>Predict flight delays using machine learning</p>
        
        <form onSubmit={handleSubmit} className="form">
          <div className="form-grid">
            <div className="form-group">
              <label>Month (1-12)</label>
              <input
                type="number"
                name="month"
                value={formData.month}
                onChange={handleInputChange}
                min="1"
                max="12"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Day of Week (1-7)</label>
              <input
                type="number"
                name="day_of_week"
                value={formData.day_of_week}
                onChange={handleInputChange}
                min="1"
                max="7"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Departure Time (HHMM)</label>
              <input
                type="number"
                name="dep_time"
                value={formData.dep_time}
                onChange={handleInputChange}
                min="0"
                max="2359"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Distance (miles)</label>
              <input
                type="number"
                name="distance"
                value={formData.distance}
                onChange={handleInputChange}
                min="50"
                max="5000"
                required
              />
            </div>
            
            <div className="form-group">
              <label>Carrier</label>
              <select
                name="carrier"
                value={formData.carrier}
                onChange={handleInputChange}
                required
              >
                {carriers.map(carrier => (
                  <option key={carrier} value={carrier}>{carrier}</option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label>Origin Airport</label>
              <select
                name="origin"
                value={formData.origin}
                onChange={handleInputChange}
                required
              >
                {airports.map(airport => (
                  <option key={airport} value={airport}>{airport}</option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label>Destination Airport</label>
              <select
                name="dest"
                value={formData.dest}
                onChange={handleInputChange}
                required
              >
                {airports.map(airport => (
                  <option key={airport} value={airport}>{airport}</option>
                ))}
              </select>
            </div>
          </div>
          
          <button type="submit" disabled={loading} className="predict-btn">
            {loading ? 'Predicting...' : 'Predict Delay'}
          </button>
        </form>
        
        {prediction && (
          <div className="result">
            {prediction.error ? (
              <div className="error">
                <h3>Error</h3>
                <p>{prediction.error}</p>
              </div>
            ) : (
              <div className={`prediction ${prediction.is_delayed ? 'delayed' : 'on-time'}`}>
                <h3>Prediction Result</h3>
                <div className="result-details">
                  <p><strong>Status:</strong> {prediction.is_delayed ? 'Likely Delayed' : 'On Time'}</p>
                  <p><strong>Delay Probability:</strong> {(prediction.delay_probability * 100).toFixed(1)}%</p>
                  <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;