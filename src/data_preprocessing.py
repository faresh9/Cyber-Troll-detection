import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
import emoji

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(file_path):
    """Load the Twitter data from CSV"""
    df = pd.read_csv(file_path)
    # Keep only English tweets
    df = df[df['Language'] == 'en']
    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)
    return df

def clean_text(text):
    """Clean and preprocess tweet text"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Replace emojis with their descriptions
        text = emoji.demojize(text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Extract hashtags but remove the # symbol
        hashtags = re.findall(r'#(\w+)', text)
        # Remove the hashtags from the text
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

def extract_features(df):
    """Extract additional features from the data"""
    # Text length
    df['text_length'] = df['Tweet_Content'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
    
    # Count of hashtags
    df['hashtag_count'] = df['Tweet_Content'].apply(
        lambda x: len(re.findall(r'#\w+', str(x))) if isinstance(x, str) else 0
    )
    
    # Count of mentions
    df['mention_count'] = df['Tweet_Content'].apply(
        lambda x: len(re.findall(r'@\w+', str(x))) if isinstance(x, str) else 0
    )
    
    # Uppercase ratio (possible indicator of shouting/aggression)
    df['uppercase_ratio'] = df['Tweet_Content'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper())/len(str(x)) if isinstance(x, str) and len(str(x)) > 0 else 0
    )
    
    # Check for controversial keywords (customize this list based on your domain)
    troll_keywords = ['terrorist', 'nazi', 'evil', 'kill', 'murder', 'destroy', 'genocide', 'apartheid']
    df['controversial_keyword_count'] = df['Tweet_Content'].apply(
        lambda x: sum(str(x).lower().count(word) for word in troll_keywords) if isinstance(x, str) else 0
    )
    
    # Engagement features that may indicate viral/controversial content
    df['engagement_score'] = df['Reply_Count'] + df['Repost_Count'] + df['Like_Count']
    
    # Verified status as a numeric feature
    df['is_verified'] = df['Verified_Status'].apply(lambda x: 1 if x == 'True' else 0)
    
    return df

def create_troll_labels(df):
    """
    Create labels for trolling detection (this is a simplified heuristic approach)
    You would ideally have human-labeled data or a more sophisticated approach
    """
    # This is a simplified labeling approach - in reality you'd want human labeling or better heuristics
    # Criteria: high controversial keywords, high uppercase ratio, and relatively high engagement
    
    # Normalize the features for scoring
    df['norm_controversial'] = df['controversial_keyword_count'] / df['controversial_keyword_count'].max()
    df['norm_uppercase'] = df['uppercase_ratio'] / df['uppercase_ratio'].max() 
    df['norm_engagement'] = np.log1p(df['engagement_score']) / np.log1p(df['engagement_score']).max()
    
    # Create a simple troll score
    df['troll_score'] = (df['norm_controversial'] * 0.5 + 
                         df['norm_uppercase'] * 0.3 + 
                         df['norm_engagement'] * 0.2)
    
    # Label as potential troll if score is above threshold
    # This threshold would need tuning
    threshold = df['troll_score'].quantile(0.75)  # Top 25% as trolls for balanced dataset
    df['troll_label'] = (df['troll_score'] > threshold).astype(int)
    
    return df

def prepare_dataset(df):
    """Prepare the final dataset for the model"""
    # Clean the tweet content
    df['cleaned_text'] = df['Tweet_Content'].apply(clean_text)
    
    # Extract features
    df = extract_features(df)
    
    # Create labels
    df = create_troll_labels(df)
    
    # Select relevant columns for modeling
    features = [
        'cleaned_text', 'text_length', 'hashtag_count', 'mention_count',
        'uppercase_ratio', 'controversial_keyword_count', 'engagement_score',
        'is_verified', 'troll_label'
    ]
    
    return df[features]

def main():
    # File path to your data
    file_path = '../data/Twitter_Advanced_Search_Scraper.csv'
    
    # Load and process data
    df = load_data(file_path)
    processed_df = prepare_dataset(df)
    
    # Save the processed data
    processed_df.to_csv('../data/processed_twitter_data.csv', index=False)
    
    # Split into train and test sets
    X = processed_df.drop('troll_label', axis=1)
    y = processed_df['troll_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save train and test sets
    pd.concat([X_train, y_train], axis=1).to_csv('../data/train_data.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('../data/test_data.csv', index=False)
    
    print(f"Processing complete. Generated {sum(y)} potential troll instances out of {len(y)} total records.")
    print(f"Processed data saved to '../data/processed_twitter_data.csv'")
    print(f"Train/test splits saved to '../data/train_data.csv' and '../data/test_data.csv'")

if __name__ == "__main__":
    main()