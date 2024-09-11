# ECG Signal Classification using CNN with K-Fold Cross-Validation

This project aims to classify ECG signals using a Convolutional Neural Network (CNN) with K-Fold Cross-Validation. The dataset consists of ECG signal recordings which are processed and segmented into fixed-length samples for training and evaluation. The code also incorporates data augmentation and regularization techniques to improve model performance.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Functions](#functions)
- [Results](#results)
- [References](#references)

Get the data from this link: https://physionet.org/content/chfdb/1.0.0/ and put into content. 

## Installation

To run this project, you will need to install the following dependencies:

```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow wfdb
Ensure you have all required ECG files (both .dat and .hea formats) in the /content/ folder for the model to process the dataset.

Project Structure
bash
Copy code
.
├── README.md            # You are here
├── main.py              # Main script to execute the program
└── /content/            # Folder containing ECG signal files (.dat and .hea)
Usage
To run the code, simply execute the main.py script:

bash
Copy code
python main.py
Dataset Preparation
The code expects ECG signal files in the /content/ directory. The load_ecg_dataset function processes and segments the ECG signals for use in training the model.

Model Architecture
A Convolutional Neural Network (CNN) is used for binary classification (normal vs. disease). The architecture includes:

Conv1D and MaxPooling layers
Dropout regularization
Fully connected layers
Sigmoid activation for binary output
Cross-Validation
K-Fold Cross-Validation (with n_splits=5) is applied to train the model and evaluate its performance across multiple folds.

Functions
process_single_ecg(file_base_path, segment_length=5000): Processes individual ECG signals, segments them into fixed lengths.
load_ecg_dataset(files, segment_length=5000): Loads and prepares the dataset for training.
build_cnn_model(input_shape): Builds the CNN architecture for ECG classification.
plot_training_history(history, fold): Plots training and validation accuracy/loss over epochs.
evaluate_model_performance(y_test, y_pred, fold): Generates confusion matrix and classification report.
plot_roc_curve(y_test, y_pred_prob, fold): Plots the ROC curve and calculates AUC for model performance.
Results
Throughout training, several metrics and plots are generated:

Confusion Matrix: Visual representation of true vs predicted labels.
ROC Curve: Illustrates the model's true positive and false positive rates.
Training History: Plots showing accuracy and loss over epochs.
Each fold of the K-Fold cross-validation will produce these outputs, providing insights into model performance across different data splits.

References
WFDB Toolbox for reading ECG files.
Keras for building the CNN model.
Scikit-learn for cross-validation and metrics.
