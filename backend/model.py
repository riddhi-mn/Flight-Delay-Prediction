import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
import os

class FlightDelayModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def preprocess_data(self, df):
        """Clean and preprocess the dataset"""
        # Remove rows with missing values
        df_clean = df.dropna()
        
        # Create delay binary target (1 if delayed, 0 if not)
        df_clean['is_delayed'] = (df_clean['dep_delay'] > 15).astype(int)
        
        # Select relevant features
        feature_cols = ['month', 'day_of_week', 'dep_time', 'distance', 
                       'carrier', 'origin', 'dest']
        
        # Handle categorical variables
        categorical_cols = ['carrier', 'origin', 'dest']
        for col in categorical_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le
        
        # Extract features and target
        available_features = [col for col in feature_cols if col in df_clean.columns]
        X = df_clean[available_features]
        y = df_clean['is_delayed']
        
        self.feature_columns = available_features
        return X, y
    
    def build_model(self, input_dim):
        """Build feedforward neural network"""
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, csv_path):
        """Train the neural network model"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(csv_path)
        
        # Handle different possible column names
        if 'dep_delay' not in df.columns and 'DepDelay' in df.columns:
            df['dep_delay'] = df['DepDelay']
        if 'carrier' not in df.columns and 'UniqueCarrier' in df.columns:
            df['carrier'] = df['UniqueCarrier']
        if 'origin' not in df.columns and 'Origin' in df.columns:
            df['origin'] = df['Origin']
        if 'dest' not in df.columns and 'Dest' in df.columns:
            df['dest'] = df['Dest']
        if 'distance' not in df.columns and 'Distance' in df.columns:
            df['distance'] = df['Distance']
        if 'dep_time' not in df.columns and 'DepTime' in df.columns:
            df['dep_time'] = df['DepTime']
        if 'day_of_week' not in df.columns and 'DayOfWeek' in df.columns:
            df['day_of_week'] = df['DayOfWeek']
        if 'month' not in df.columns and 'Month' in df.columns:
            df['month'] = df['Month']
        
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        print("Building neural network...")
        self.model = self.build_model(X_train_scaled.shape[1])
        
        print("Training model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate
        y_pred = (self.model.predict(X_test_scaled) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model components
        self.save_model()
        
        return accuracy
    
    def predict(self, flight_data):
        """Predict delay probability for a single flight"""
        if self.model is None:
            self.load_model()
        
        # Convert to DataFrame
        df = pd.DataFrame([flight_data])
        
        # Apply same preprocessing
        for col, le in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select and scale features
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prob = self.model.predict(X_scaled)[0][0]
        return float(prob)
    
    def save_model(self):
        """Save trained model and preprocessors"""
        self.model.save('flight_delay_model.h5')
        
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        with open('feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
    
    def load_model(self):
        """Load trained model and preprocessors"""
        if os.path.exists('flight_delay_model.h5'):
            self.model = tf.keras.models.load_model('flight_delay_model.h5')
            
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
                
            with open('feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            return True
        return False