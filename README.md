# Speaker-Classification-KNN

This repository contains a project focused on speaker classification using supervised learning techniques. The project involves the following steps:

- **Data Preparation**: Splitting the dataset into training and testing sets, and reading the audio data.
- **Feature Extraction**: Using audio features like zero-crossing rate, short-time energy, pitch, and Mel-Frequency Cepstral Coefficients (MFCC) to extract useful information from audio data.
- **Voice Activity Detection**: Filtering features to isolate voiced speech using energy and zero-crossing rate thresholds.
- **Feature Normalization**: Standardizing features to improve classifier performance.
- **Model Training**: Implementing and training a K-Nearest Neighbor (KNN) classifier and a Naive Bayes classifier.
- **Cross-Validation**: Evaluating the KNN classifier using k-fold cross-validation to assess its performance.
- **Model Evaluation**: Testing the classifiers on a separate test set and visualizing the results using confusion matrices.

**Keywords**: Voice Classification, Supervised Learning, KNN, Naive Bayes, MFCC, Pitch, Audio Feature Extraction

## Changelog

### 2024-06-03
- Implemented speaker classification using KNN and Naive Bayes algorithms:
  - Added file selection and audio data loading functionality.
  - Applied feature extraction using zero-crossing rate, short-time energy, pitch, and Mel-Frequency Cepstral Coefficients (MFCC).
  - Performed voice activity detection using energy and zero-crossing rate thresholds.
  - Normalized features for improved classifier performance.
  - Trained a K-Nearest Neighbor (KNN) classifier and evaluated its performance with k-fold cross-validation.
  - Tested the KNN classifier on a test set and visualized results with a confusion matrix.
  - Trained and tested a Naive Bayes classifier, and visualized results with a confusion matrix.
