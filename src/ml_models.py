from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class NetflixMLModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for ML models"""
        features = {}
        
        # Text features from description
        if 'description' in df.columns:
            features['description_vec'] = self.vectorizer.fit_transform(df['description'].fillna(''))
            
        # Numerical features
        numerical_features = []
        if 'duration' in df.columns:
            numerical_features.append(df['duration'].fillna(0))
            
        if numerical_features:
            features['numerical'] = self.scaler.fit_transform(
                pd.concat(numerical_features, axis=1))
            
        return features
        
    def cluster_content(self, features, n_clusters=5):
        """Cluster Netflix content based on features"""
        if 'description_vec' not in features:
            return None
            
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(features['description_vec'])
        return clusters
        
    def reduce_dimensions(self, features, n_components=2):
        """Reduce dimensions for visualization"""
        if 'description_vec' not in features:
            return None
            
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features['description_vec'].toarray())
        return reduced_features
        
    def get_similar_content(self, df, title, n_similar=5):
        """Find similar content based on description"""
        if 'description' not in df.columns or 'title' not in df.columns:
            return None
            
        if title not in df['title'].values:
            return None
            
        content_idx = df[df['title'] == title].index[0]
        description_vec = self.vectorizer.transform(df['description'].fillna(''))
        
        # Calculate similarities
        similarities = description_vec[content_idx] @ description_vec.T.toarray()
        similar_indices = similarities.argsort()[-n_similar-1:-1][::-1]
        
        return df.iloc[similar_indices][['title', 'description']]