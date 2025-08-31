import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib # Used for saving the model

def train_and_save_model():
    """Trains the sentiment analysis model and saves it to a file."""
    print("Loading data...")
    # Make sure the path is correct for where this script is running
    df = pd.read_csv('IMDB Dataset.csv')

    # Using a larger sample for the final model for better performance
    df_sample = df.sample(n=25000, random_state=42)
    X = df_sample['review']
    y = df_sample['sentiment']

    # Define the final model pipeline
    print("Creating model pipeline...")
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
        ('classifier', LogisticRegression(C=5, solver='liblinear'))
    ])

    # Train the model on the full sample data
    print("Training model...")
    model_pipeline.fit(X, y)

    # Save the trained model pipeline to a file
    print("Saving model pipeline to sentiment_model.pkl...")
    joblib.dump(model_pipeline, 'sentiment_model.pkl')
    print("Model training complete and saved! ✅")

if __name__ == '__main__':
    train_and_save_model()