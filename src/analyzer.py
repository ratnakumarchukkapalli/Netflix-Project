import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class NetflixAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def plot_genre_distribution(self, top_n=10):
        """Plot distribution of genres"""
        if 'listed_in' not in self.df.columns:
            print("No genre information available")
            return
            
        genres = self.df['listed_in'].str.split(',').explode().str.strip()
        genre_counts = genres.value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=genre_counts.values, y=genre_counts.index)
        plt.title(f'Top {top_n} Genres on Netflix')
        plt.xlabel('Count')
        plt.ylabel('Genre')
        plt.tight_layout()
        return plt
        
    def plot_yearly_additions(self):
        """Plot content additions over years"""
        if 'date_added' not in self.df.columns:
            print("No date information available")
            return
            
        yearly_additions = self.df['date_added'].dt.year.value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        yearly_additions.plot(kind='line', marker='o')
        plt.title('Content Added to Netflix by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Titles Added')
        plt.grid(True)
        plt.tight_layout()
        return plt
        
    def get_content_duration_stats(self):
        """Analyze content duration statistics"""
        if 'duration' not in self.df.columns:
            return "No duration information available"
            
        stats = {
            'mean_duration': self.df['duration'].mean(),
            'median_duration': self.df['duration'].median(),
            'std_duration': self.df['duration'].std(),
            'min_duration': self.df['duration'].min(),
            'max_duration': self.df['duration'].max()
        }
        return stats