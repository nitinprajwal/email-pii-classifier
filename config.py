# config.py - Developed by nitinprajwal
"""
Configuration settings for the Email Classification and PII Masking application.

This file centralizes configuration parameters such as file paths, model locations,
and logging settings to make the application more maintainable and configurable.
All paths are constructed dynamically based on the project's root directory.
"""

import os

# Project Root Directory
# Dynamically determines the absolute path to the project's root directory.
# This ensures that file paths are correctly resolved regardless of where the
# application is run from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Model Paths
# Path to the pre-trained email classification model file.
# The model is expected to be a .joblib file located in the project root.
CLASSIFICATION_MODEL_PATH = os.path.join(PROJECT_ROOT, "email_classifier.joblib")

# Logging Configuration
# Defines the minimum severity level for log messages to be recorded.
# Common levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
LOG_LEVEL = "INFO"

# Defines the format string for log messages.
# This format includes timestamp, logger name, log level, and the message itself.
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

