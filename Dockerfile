# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# The en_core_web_sm model might be installed via requirements.txt if specified with a URL.
# Running `python -m spacy download en_core_web_sm` ensures it's available.
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy the rest of the application code into the container
# This includes app.py, pii_masking.py, email_classifier.joblib, etc.
# Ensure .dockerignore is used to exclude unnecessary files.
COPY . .

# Hugging Face Spaces will use the Procfile to run the application.
# The Procfile should be: web: uvicorn app:app --host 0.0.0.0 --port $PORT
# The $PORT environment variable will be set by Hugging Face Spaces,
# based on the `app_port` in the README.md YAML (e.g., 7860).
