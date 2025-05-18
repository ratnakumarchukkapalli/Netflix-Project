from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class NetflixMLAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def prepare_content_features(self, titles):
        """Convert titles into feature vectors"""
        return self.vectorizer.fit_transform(titles)
    
    def find_similar_content(self, df, title, n_recommendations=5):
        """Find similar content based on title similarity"""
        # Create a matrix of title features
        title_matrix = self.prepare_content_features(df['Title'])
        
        # Find the index of the input title
        title_idx = df[df['Title'] == title].index[0]
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(title_matrix[title_idx], title_matrix)
        
        # Get indices of most similar titles (excluding the input title)
        similar_indices = similarity_scores.argsort()[0][-n_recommendations-1:-1][::-1]
        
        return df.iloc[similar_indices][['Title', 'Date']]
    
    def cluster_viewing_patterns(self, df, n_clusters=4):
        """Cluster viewing patterns based on time features"""
        # Extract time features
        time_features = pd.DataFrame({
            'hour': df['Date'].dt.hour,
            'day_of_week': df['Date'].dt.dayofweek,
            'month': df['Date'].dt.month
        })
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(time_features)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(scaled_features)
        
        return clusters
    
    def predict_next_viewing_time(self, df):
        """Predict likely viewing times based on historical patterns"""
        # Create time-based features
        viewing_patterns = pd.DataFrame({
            'hour': df['Date'].dt.hour,
            'day_of_week': df['Date'].dt.dayofweek
        })
        
        # Calculate probability distribution
        hour_probs = viewing_patterns['hour'].value_counts(normalize=True)
        day_probs = viewing_patterns['day_of_week'].value_counts(normalize=True)
        
        return {
            'most_likely_hour': hour_probs.index[0],
            'most_likely_day': day_probs.index[0],
            'hour_probabilities': hour_probs,
            'day_probabilities': day_probs
        }

    def analyze_binge_patterns(self, df):
        """Analyze and predict binge-watching behavior"""
        # Calculate time differences between consecutive views
        df = df.sort_values('Date')
        time_diffs = df['Date'].diff()
        
        # Define binge watching (shows watched within 24 hours)
        binge_threshold = pd.Timedelta(hours=24)
        binge_sessions = time_diffs <= binge_threshold
        
        # Analyze binge patterns
        binge_shows = df[binge_sessions]['Title'].value_counts()
        
        return {
            'binge_ratio': binge_sessions.mean(),
            'top_binged_shows': binge_shows.head(),
            'average_session_length': len(binge_sessions[binge_sessions].index)
        }