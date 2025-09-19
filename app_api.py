import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Define the path to the dataset and model
DATA_PATH = 'data/Crop_recommendation.csv'
MODEL_PATH = 'crop_model.pkl'

# --- Model Training and Loading Logic ---
def train_and_save_model():
    """Trains the model and saves it to a file."""
    print("No pre-trained model found. Training a new model...")
    
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    
    # Define features (X) and target (y)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Split the data into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(model, MODEL_PATH)
    print(f"Model training complete. Saved as {MODEL_PATH}")
    return model

def load_model():
    """Loads the model from a file or trains a new one."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save_model()

# Load the model when the application starts
model = load_model()

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for crop recommendation.
    It expects a POST request with a JSON body.
    """
    try:
        data = request.json
        
        required_keys = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Missing one or more required parameters.'}), 400

        input_data = pd.DataFrame([data])
        
        prediction = model.predict(input_data)
        
        return jsonify({'recommended_crop': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    """A simple welcome message to confirm the API is running."""
    return "API is running!"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
