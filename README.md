# Grammar Scoring Engine

This project implements a machine learning model to predict grammar scores (0-5) from spoken audio samples. The model analyzes audio features to assess grammatical accuracy and structure in speech.

### Audio files: https://drive.google.com/file/d/1gA48Dv5m4KcqLSLT-tBwwmDQ419zjBi9/view?usp=drive_link

## Project Structure

```
.
├── dataset/
│   ├── test.csv          # Test set metadata
│   ├── train.csv         # Training set with labels
│   └── sample_submission.csv
├── audios_test/          # Test audio files
├── audios_train/         # Training audio files
├── grammar_scoring_engine.ipynb  # Main notebook
├── requirements.txt      # Project dependencies
├── submission.csv       # final sample submission file
├── test_output.csv      # final submission file
├── voice.py             # testing the audio transcription
└── README.md            # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Model Architecture

The grammar scoring model uses a deep neural network that processes audio features including:
- Mel-frequency cepstral coefficients (MFCCs)
- Spectral features (centroids, rolloff)
- Rhythm features
- Zero crossing rate

## Grammar Score Rubric

| Score | Description |
|-------|-------------|
| 1 | Poor grammar control, incomplete/memorized sentences |
| 2 | Limited understanding, basic mistakes |
| 3 | Decent grammar with syntax/structure errors |
| 4 | Strong understanding, minor errors |
| 5 | High grammatical accuracy, complex structure handling |

## Evaluation

The model is evaluated based on:
- Mean Squared Error (MSE)
- R-squared (R²) score
- Distribution of predictions vs. actual scores 
