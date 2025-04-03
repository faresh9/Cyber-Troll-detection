# Cyber Trolling Detection in Israel-Palestine Conflict Discourse

## Overview

This project aims to detect cyber trolling in social media discussions related to the Israel-Palestine conflict. Using natural language processing and deep learning techniques, the system analyzes Twitter/X data to identify harmful speech patterns, disinformation, and coordinated inauthentic behavior.

## Project Structure

```
Cyber-Troll-detection/
│
├── data/                          # Data directory
│   ├── Twitter_Advanced_Search_Scraper.csv   # Raw data
│   ├── processed_twitter_data.csv            # Cleaned data
│   ├── train_data.csv                        # Training split
│   ├── test_data.csv                         # Testing split
│   └── manual_labeling_sample.csv            # Sample for human annotation
│
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data cleaning and preparation
│   ├── manual_labeling.py         # Script for creating labeling samples
│   ├── model.py                   # Model architecture and training
│   └── utils.py                   # Utility functions
│
├── models/                        # Trained models
│   ├── troll_detection_model.h5   # Saved model
│   └── tokenizer.pickle           # Saved tokenizer
│
├── notebooks/                     # Jupyter notebooks for exploration
│
└── README.md                      # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Cyber-Troll-detection.git
cd Cyber-Troll-detection

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
pandas>=1.3.0
numpy>=1.19.5
scikit-learn>=0.24.2
tensorflow>=2.6.0
nltk>=3.6.2
emoji>=1.6.1
matplotlib>=3.4.3
seaborn>=0.11.2
```

## Data

The project uses Twitter/X data collected during the October 2023 Israel-Hamas conflict. The dataset includes tweets containing keywords related to the conflict, along with metadata such as:

- Tweet content
- Author information
- Engagement metrics (likes, replies, reposts)
- Timestamp information
- Media links
- Reply connections

## Usage

### Data Preprocessing

```bash
# Process the raw Twitter data
python src/data_preprocessing.py
```

### Model Training

```bash
# Train the trolling detection model
python src/model.py
```

### Manual Labeling

```bash
# Generate a sample for manual annotation
python src/manual_labeling.py
```

## Methodology

Our approach combines NLP techniques with deep learning to detect trolling behavior:

1. **Text Cleaning**: Remove URLs, special characters, and normalize text
2. **Feature Engineering**: Extract features including:
   - Text statistics (length, uppercase ratio)
   - Engagement metrics
   - Hashtag and mention counts
   - Controversial keyword presence
3. **Model Architecture**: Bidirectional LSTM neural network with:
   - Word embeddings
   - Multiple LSTM layers
   - Dropout for regularization
4. **Evaluation**: Precision, recall, and F1-score metrics prioritizing troll detection

## Current Results

*Note: This section will be updated once model performance metrics are available.*

## Next Steps

### 1. Data Enhancement

- **Expand Data Collection**: Gather more diverse tweets across different time periods and events
- **Human Annotation**: Implement a comprehensive labeling protocol with multiple annotators
- **Label Quality**: Calculate inter-annotator agreement and resolve disagreements
- **Balanced Dataset**: Ensure appropriate representation of trolling vs. non-trolling content

### 2. Model Improvements

- **Pre-trained Language Models**: Implement transformer-based models like BERT, RoBERTa or other domain-specific LLMs
- **Multi-modal Analysis**: Incorporate image analysis for tweets containing media
- **Context Awareness**: Consider conversation threads and user history for better detection
- **Cross-lingual Support**: Extend detection capabilities to Arabic, Hebrew, and other relevant languages

### 3. Feature Development

- **Network Analysis**: Implement graph-based features to detect coordinated behavior
- **Temporal Patterns**: Analyze posting frequency and timing patterns
- **Linguistic Markers**: Develop more sophisticated linguistic features specific to online trolling
- **User Behavior**: Track consistent behavioral patterns across multiple posts

### 4. System Integration

- **Real-time Processing**: Develop a pipeline for processing and analyzing tweets in real-time
- **API Development**: Create an API for integration with other systems
- **Visualization Dashboard**: Develop an interactive dashboard to explore results
- **Alert System**: Implement notification mechanisms for detected trolling campaigns

### 5. Ethical Considerations

- **Bias Mitigation**: Regularly audit and address potential biases in the model
- **Transparency**: Document decision-making processes and model limitations
- **Privacy Protection**: Ensure responsible use of public data
- **Impact Assessment**: Evaluate the broader societal impact of the tool

### 6. Evaluation and Validation

- **Benchmark Creation**: Establish a benchmark dataset for cyber trolling in conflict discourse
- **Adversarial Testing**: Test model against sophisticated evasion techniques
- **External Validation**: Seek expert validation from conflict analysis researchers
- **Continuous Monitoring**: Establish protocols for regular performance evaluation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and researchers in the field of computational social science and conflict studies
- Special thanks to annotators who contributed to the dataset labeling
