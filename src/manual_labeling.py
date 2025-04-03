import pandas as pd
import random

def create_labeling_sample(df, sample_size=200):
    """Create a sample for manual labeling"""
    # Sample tweets for manual labeling
    sampled_df = df.sample(sample_size, random_state=42)
    
    # Add columns for manual labeling
    sampled_df['is_troll'] = None
    sampled_df['troll_category'] = None
    sampled_df['notes'] = None
    
    # Save the sample for manual labeling
    sampled_df.to_csv('../data/manual_labeling_sample.csv', index=False)
    
    return sampled_df

# Example usage
# df = pd.read_csv('../data/Twitter_Advanced_Search_Scraper.csv')
# create_labeling_sample(df)