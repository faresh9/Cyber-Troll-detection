import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

def build_text_sequence_model(max_words=10000, max_sequence_length=100):
    """Build a sequential model for text classification"""
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def prepare_text_sequences(texts, max_words=10000, max_sequence_length=100):
    """Convert text to sequences for model input"""
    # Create tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    return padded_sequences, tokenizer

def main():
    # Load training data
    train_df = pd.read_csv('../data/train_data.csv')
    test_df = pd.read_csv('../data/test_data.csv')
    
    # Prepare text sequences
    X_train_seq, tokenizer = prepare_text_sequences(train_df['cleaned_text'])
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(test_df['cleaned_text']), 
        maxlen=100
    )
    
    # Additional numerical features
    numerical_features = [
        'text_length', 'hashtag_count', 'mention_count',
        'uppercase_ratio', 'controversial_keyword_count', 
        'engagement_score', 'is_verified'
    ]
    
    X_train_numerical = train_df[numerical_features].values
    X_test_numerical = test_df[numerical_features].values
    
    # Target variable
    y_train = train_df['troll_label'].values
    y_test = test_df['troll_label'].values
    
    # Build and train model
    model = build_text_sequence_model()
    
    # Train model
    history = model.fit(
        X_train_seq, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate model
    results = model.evaluate(X_test_seq, y_test)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")
    print(f"Test Precision: {results[2]}")
    print(f"Test Recall: {results[3]}")
    
    # Save model
    model.save('../models/troll_detection_model.h5')
    
    # Save tokenizer
    import pickle
    with open('../models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()