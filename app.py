from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model when the application starts
print("Loading the trained model...")
model = joblib.load('sentiment_model.pkl')
print("Model loaded successfully.")

@app.route('/')
def home():
    """Renders the main HTML page for the user interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives review text via a POST request and returns the sentiment."""
    try:
        # Get data from the POST request
        data = request.get_json()
        review_text = data['review']
        
        # The model expects a list of texts for prediction
        prediction = model.predict([review_text])
        
        # Return the prediction as JSON
        return jsonify({'sentiment': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Starts the web server
    app.run(port=5000, debug=True)