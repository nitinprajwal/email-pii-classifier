# classification_model.py - Developed by nitinprajwal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # Optional: for evaluating model
from sklearn.metrics import classification_report # Optional: for evaluating model
import pandas as pd
import joblib
import os

# Assuming utils.py is in the same directory
from utils import load_data
# Import PII masking functionality
from pii_masking import mask_pii_details, nlp as spacy_nlp_model_for_training # Use the loaded spaCy model

from config import CLASSIFICATION_MODEL_PATH
MODEL_FILENAME = CLASSIFICATION_MODEL_PATH
DEFAULT_DATASET_PATH = "combined_emails_with_natural_pii.csv"

def train_classification_model(data_path: str = DEFAULT_DATASET_PATH, model_save_path: str = MODEL_FILENAME):
    """
    Trains the email classification model and saves it.
    Uses 'email' column for text and 'type' for category.
    """
    print(f"Starting model training with dataset: {data_path}")
    df = load_data(data_path)

    if df is None:
        print("Failed to load data. Aborting training.")
        return False

    # Preprocessing: Fill NaN in 'email' (text content) and 'type' (labels)
    df['email'] = df['email'].fillna('')
    df['type'] = df['type'].fillna('Unknown')
    df.dropna(subset=['type'], inplace=True) # Ensure labels are present

    if df.empty or df['email'].empty or df['type'].empty:
        print("Data is empty or lacks required 'email' or 'type' columns after preprocessing. Aborting training.")
        return False

    print("Applying PII masking to training data...")
    # Ensure the spaCy model is available for masking
    if spacy_nlp_model_for_training is None:
        print("Warning: spaCy model not loaded in pii_masking. Training will use regex-only masked data.")

    # Mask PII in the training data
    # This can be slow for large datasets; consider optimizations if needed
    masked_emails = []
    for i, email_text in enumerate(df['email']):
        if pd.isna(email_text):
            masked_emails.append("") # Handle potential NaN after fillna('') if any slip through
            continue
        masked_text, _ = mask_pii_details(str(email_text), nlp_model=spacy_nlp_model_for_training)
        masked_emails.append(masked_text)
        if (i + 1) % 100 == 0:
            print(f"Masked {i+1}/{len(df['email'])} emails for training...")
    
    df['masked_email_for_training'] = masked_emails
    print("PII masking for training data complete.")

    X = df['masked_email_for_training']
    y = df['type']

    # Optional: Split data for evaluation (not strictly required by assignment but good practice)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create a pipeline: TF-IDF Vectorizer -> Multinomial Naive Bayes
    # You can experiment with other models like SVM, Logistic Regression, or even simple Transformers.
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1,2))),
        ('clf', MultinomialNB(alpha=0.1)), # Alpha is a smoothing parameter for Naive Bayes
    ])

    print("Training the model...")
    # model.fit(X_train, y_train) # If using train_test_split
    model.fit(X, y) # Train on full dataset as per typical assignment flow unless evaluation is separate
    print("Model training complete.")

    # Optional: Evaluate the model
    # print("\nModel Evaluation on Test Set:")
    # predictions = model.predict(X_test)
    # print(classification_report(y_test, predictions))

    try:
        joblib.dump(model, CLASSIFICATION_MODEL_PATH)
        print(f"Model saved to {CLASSIFICATION_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_classification_model(model_path: str = CLASSIFICATION_MODEL_PATH):
    """
    Loads the trained classification model.
    """
    if not os.path.exists(CLASSIFICATION_MODEL_PATH):
        print(f"Error: Model file not found at {CLASSIFICATION_MODEL_PATH}. Train the model first or ensure path is correct.")
        print(f"Attempting to train a new model with default dataset: {DEFAULT_DATASET_PATH}")
        success = train_classification_model(data_path=DEFAULT_DATASET_PATH, model_save_path=CLASSIFICATION_MODEL_PATH)
        if not success:
            print("Failed to train a new model. Cannot load model.")
            return None
        # If training was successful, the model file should now exist.
    
    try:
        model = joblib.load(CLASSIFICATION_MODEL_PATH)
        print(f"Model loaded successfully from {CLASSIFICATION_MODEL_PATH}")
        return model
    except FileNotFoundError:
        # This case should be handled by the os.path.exists check and auto-train attempt now.
        print(f"Error: Model file not found at {CLASSIFICATION_MODEL_PATH} even after attempting to train.")
        return None
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def classify_email_category(masked_email_text: str, model):
    """
    Classifies the masked email text into a category.
    """
    if model is None:
        print("Error: Classification model not loaded.")
        # Fallback category or raise an error, as per application requirements
        return "Error: Model not available"
    try:
        # The model expects a list or iterable of texts
        prediction = model.predict([masked_email_text])
        return prediction[0]
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Error: Classification failed"

if __name__ == "__main__":
    print("Running classification_model.py script...")
    # Train the model using the provided dataset
    # This will save the model as 'email_classifier.joblib' in the root directory
    training_successful = train_classification_model(data_path=DEFAULT_DATASET_PATH, model_save_path=MODEL_FILENAME)

    if training_successful:
        print("\n--- Testing loaded model ---_model")
        # Load the just-trained model
        loaded_model = load_classification_model(MODEL_FILENAME)
        if loaded_model:
            sample_emails_for_testing = [
                ("Subject: Urgent - Server down! Our main application server is not responding. We need immediate assistance.", "Incident"),
                ("Subject: Password Reset Request. Hi, I forgot my password and need to reset it. My username is testuser.", "Request"),
                ("Subject: Inquiry about new billing plans. Could you please provide more information on your enterprise billing options?", "Request"),
                ("Subject: System Update Notification for 2023-01-15. We will be performing scheduled maintenance.", "Change"),
                ("Subject: Recurring login issue. I've been unable to login for the past three days, the error says 'invalid credentials' but I am sure they are correct.", "Problem"),
            ]
            print("\nClassifying sample emails:")
            for email_text, expected_category in sample_emails_for_testing:
                # For testing the endpoint, the API will handle masking. 
                # For this direct model test, we should simulate that by masking first.
                print(f"\nOriginal sample for testing: {email_text[:60]}...")
                masked_sample_text, _ = mask_pii_details(email_text, nlp_model=spacy_nlp_model_for_training) # Use the same nlp model
                print(f"Masked sample for testing: {masked_sample_text[:60]}...")
                category = classify_email_category(masked_sample_text, loaded_model)
                print(f"-> Predicted: {category} (Expected: {expected_category})")
    else:
        print("Model training failed. Cannot proceed with testing.")

