"""
Cardiovascular Disease Prediction System
Complete Flask Application
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import json
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Sample data for demonstration
def load_sample_data():
    """Load sample cardiovascular data"""
    # In real implementation, load from CSV
    data = {
        'age': [50, 45, 60, 35, 55],
        'ap_hi': [140, 130, 160, 120, 150],
        'ap_lo': [90, 85, 100, 80, 95],
        'cholesterol': [2, 1, 3, 1, 2],
        'gluc': [1, 1, 2, 1, 1],
        'smoke': [0, 0, 1, 0, 0],
        'alco': [0, 0, 1, 0, 0],
        'active': [1, 1, 0, 1, 1],
        'cardio': [1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

class CardiovascularModel:
    """Mock ML model for cardiovascular disease prediction"""
    
    def __init__(self):
        self.feature_importance = {
            'age': 0.25,
            'ap_hi': 0.22,
            'ap_lo': 0.18,
            'cholesterol': 0.15,
            'gluc': 0.10,
            'smoke': 0.05,
            'alco': 0.03,
            'active': 0.02
        }
    
    def predict(self, features):
        """Predict cardiovascular disease risk"""
        # Simplified risk calculation
        risk_score = 0
        
        # Age contribution
        age = features['age']
        risk_score += (age / 100) * self.feature_importance['age']
        
        # Blood pressure contribution
        risk_score += (features['ap_hi'] / 200) * self.feature_importance['ap_hi']
        risk_score += (features['ap_lo'] / 150) * self.feature_importance['ap_lo']
        
        # Cholesterol (1=normal, 2=above normal, 3=well above)
        risk_score += (features['cholesterol'] / 3) * self.feature_importance['cholesterol']
        
        # Glucose
        risk_score += (features['gluc'] / 3) * self.feature_importance['gluc']
        
        # Lifestyle factors
        risk_score += features['smoke'] * self.feature_importance['smoke']
        risk_score += features['alco'] * self.feature_importance['alco']
        risk_score += (1 - features['active']) * self.feature_importance['active']
        
        # Normalize to 0-100%
        probability = min(100, max(0, risk_score * 100))
        
        # Binary prediction
        prediction = 1 if probability > 50 else 0
        
        return prediction, probability
    
    def get_recommendations(self, probability, features):
        """Get health recommendations based on risk"""
        recommendations = []
        
        if probability > 70:
            recommendations.append("🔴 Consult a healthcare professional immediately")
            recommendations.append("📅 Schedule comprehensive cardiac screening")
            recommendations.append("💊 Monitor blood pressure daily")
        elif probability > 30:
            recommendations.append("🟡 Schedule regular health check-ups")
            recommendations.append("🏃 Maintain regular physical activity")
            recommendations.append("🥗 Follow heart-healthy diet")
        else:
            recommendations.append("🟢 Continue healthy lifestyle habits")
            recommendations.append("⚖️ Maintain healthy weight")
            recommendations.append("😊 Manage stress levels")
        
        # Specific recommendations based on features
        if features['cholesterol'] > 1:
            recommendations.append("🍳 Reduce saturated fat intake")
        
        if features['ap_hi'] > 140:
            recommendations.append("🧂 Reduce salt consumption")
        
        if features['smoke'] == 1:
            recommendations.append("🚭 Consider smoking cessation programs")
        
        if features['active'] == 0:
            recommendations.append("🚶 Start with 30-minute daily walks")
        
        return recommendations

# Initialize model
model = CardiovascularModel()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html', 
                         title="CardioPredict AI",
                         current_date=datetime.now().strftime("%B %d, %Y"))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction endpoint"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    elif request.method == 'POST':
        try:
            # Extract form data
            data = request.form
            
            features = {
                'age': float(data.get('age', 50)),
                'height': float(data.get('height', 170)),
                'weight': float(data.get('weight', 70)),
                'ap_hi': float(data.get('ap_hi', 120)),
                'ap_lo': float(data.get('ap_lo', 80)),
                'cholesterol': int(data.get('cholesterol', 1)),
                'gluc': int(data.get('gluc', 1)),
                'smoke': int(data.get('smoke', 0)),
                'alco': int(data.get('alco', 0)),
                'active': int(data.get('active', 1))
            }
            
            # Calculate BMI
            bmi = features['weight'] / ((features['height']/100) ** 2)
            features['bmi'] = round(bmi, 2)
            
            # Make prediction
            prediction, probability = model.predict(features)
            
            # Get recommendations
            recommendations = model.get_recommendations(probability, features)
            
            # Determine risk level
            if probability < 30:
                risk_level = "Low Risk"
                risk_color = "success"
            elif probability < 70:
                risk_level = "Medium Risk"
                risk_color = "warning"
            else:
                risk_level = "High Risk"
                risk_color = "danger"
            
            # Prepare response
            response = {
                'success': True,
                'prediction': int(prediction),
                'probability': round(probability, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'bmi': features['bmi'],
                'recommendations': recommendations,
                'features': features
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    # Load sample data
    df = load_sample_data()
    
    # Calculate statistics
    stats = {
        'total_samples': len(df),
        'disease_prevalence': df['cardio'].mean() * 100,
        'avg_age': df['age'].mean(),
        'avg_bp': df['ap_hi'].mean(),
        'high_risk_count': len(df[df['age'] > 50]),
        'smoking_rate': df['smoke'].mean() * 100
    }
    
    return render_template('analytics.html',
                         stats=stats,
                         feature_importance=model.feature_importance)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Cardiovascular Disease Prediction',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Cardiovascular Disease Prediction System")
    print("=" * 60)
    print("Starting Flask application...")
    print("Server running on: http://localhost:5000")
    print("Available routes:")
    print("  /              - Home page")
    print("  /predict       - Risk assessment")
    print("  /analytics     - Analytics dashboard")
    print("  /api/health    - Health check API")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)