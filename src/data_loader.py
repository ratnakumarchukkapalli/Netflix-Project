import pandas as pd
import numpy as np
from pathlib import Path

class NetflixDataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        
    def load_data(self):
        """Load Netflix viewing data"""
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df):
        """Basic preprocessing of Netflix viewing data"""
        if df is None:
            return None
        
        # Convert date columns to datetime
        if 'date_added' in df.columns:
            df['date_added'] = pd.to_datetime(df['date_added'])
        
        # Handle missing values
        df = df.fillna({'duration': 0, 'rating': 'Not Rated'})
        
        return df
    
    def get_basic_stats(self, df):
        """Calculate basic viewing statistics"""
        if df is None:
            return None
            
        stats = {
            'total_entries': len(df),
            'unique_titles': df['title'].nunique() if 'title' in df.columns else 0,
            'unique_genres': self._count_unique_genres(df) if 'listed_in' in df.columns else 0,
            'time_span': self._get_time_span(df)
        }
        return stats
    
    def _count_unique_genres(self, df):
        """Helper method to count unique genres"""
        if 'listed_in' not in df.columns:
            return 0
        genres = df['listed_in'].str.split(',').explode().str.strip()
        return genres.nunique()
    
    def _get_time_span(self, df):
        """Helper method to get the time span of the dataset"""
        if 'date_added' not in df.columns:
            return "No date information available"
        return f"{df['date_added'].min():%Y-%m-%d} to {df['date_added'].max():%Y-%m-%d}"