# feature_extractor.py - Advanced Text Feature Extractor for Email Classification
"""
Advanced text feature extractor for email classification.
This module contains the AdvancedTextFeatureExtractor class that extracts
linguistic and semantic features from email text.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AdvancedTextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Advanced feature extractor for email text analysis.
    Extracts multiple linguistic and semantic features.
    """
    
    def __init__(self):
        self.urgency_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'priority']
        self.sentiment_positive = ['good', 'great', 'excellent', 'please', 'thank', 'appreciate']
        self.sentiment_negative = ['bad', 'terrible', 'awful', 'problem', 'issue', 'error', 'fail']
        self.request_keywords = ['request', 'need', 'want', 'could', 'would', 'please', 'help']
        self.technical_keywords = ['server', 'database', 'api', 'system', 'code', 'bug', 'fix']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_lower = text.lower()
            
            # Basic text statistics
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            # Urgency indicators
            urgency_score = sum([text_lower.count(word) for word in self.urgency_keywords])
            
            # Sentiment indicators
            positive_score = sum([text_lower.count(word) for word in self.sentiment_positive])
            negative_score = sum([text_lower.count(word) for word in self.sentiment_negative])
            
            # Email type indicators
            request_score = sum([text_lower.count(word) for word in self.request_keywords])
            technical_score = sum([text_lower.count(word) for word in self.technical_keywords])
            
            # Subject line analysis
            has_subject = 1 if 'subject:' in text_lower else 0
            
            # Question indicators
            question_count = text.count('?')
            
            # Exclamation indicators
            exclamation_count = text.count('!')
            
            # Capitalization patterns
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            
            # Time-related keywords
            time_keywords = ['today', 'tomorrow', 'yesterday', 'now', 'soon', 'later']
            time_score = sum([text_lower.count(word) for word in time_keywords])
            
            # Feature vector
            feature_vector = [
                word_count, char_count, sentence_count,
                urgency_score, positive_score, negative_score,
                request_score, technical_score, has_subject,
                question_count, exclamation_count, caps_ratio, time_score
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
