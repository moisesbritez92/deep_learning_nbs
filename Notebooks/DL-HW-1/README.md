# scRNA-seq Cell Type Classification - Complete Solution

## Overview

This directory contains a complete solution for Deep Learning Homework 1: Classification of cell types using single-cell RNA sequencing (scRNA-seq) data.

## Files

- `train.pkl` - Training dataset (cells x genes matrix with labels)
- `test.pkl` - Test dataset (cells x genes matrix with labels)  
- `ADI_scRNA_classification.ipynb` - Original assignment template
- `scRNA_Complete_Homework_Solution.ipynb` - **Complete solution notebook**
- `ADI_scRNA_classification.md` - Assignment description

## Complete Solution Notebook

The `scRNA_Complete_Homework_Solution.ipynb` notebook provides a comprehensive solution with:

### 1. Data Loading and Preparation
- Import necessary libraries (NumPy, Pandas, Matplotlib, Seaborn, TensorFlow)
- Load train and test datasets
- Extract common genes between datasets
- Explore data dimensions and class distribution

### 2. Theoretical Questions (Q1-Q6)
- Q1: Problem type (multiclass classification)
- Q2: Input size (number of genes/features)
- Q3: Output layer neurons (number of classes)
- Q4: Output activation function (softmax)
- Q5: Label modification (LabelEncoder + one-hot encoding)
- Q6: Loss function (categorical cross-entropy)

### 3. Label Processing
- Convert textual labels to integers using LabelEncoder
- Convert to one-hot encoding using to_categorical
- Prepare train/validation split (80/20)

### 4. Model Architectures

Three models demonstrating different fitting behaviors:

#### Underfit Model
- Architecture: 1 hidden layer with 16 neurons
- Reason for underfit: Insufficient capacity
- Expected behavior: Low accuracy on both train and validation

#### OK Model  
- Architecture: 3 hidden layers (128-64-32 neurons)
- Reason for good fit: Balanced capacity
- Expected behavior: Good accuracy on both train and validation

#### Overfit Model
- Architecture: 5 hidden layers (512-512-256-256-128 neurons)
- Reason for overfit: Excessive capacity
- Expected behavior: Very high train accuracy, lower validation accuracy

### 5. Training Without Regularization
- Training code for all three models
- Visualization of training curves (loss and accuracy)
- Analysis of train/validation gap

### 6. Regularization Techniques

Applied to the overfit model:

#### L2 Regularization (Weight Decay)
- Implementation using kernel_regularizer
- Lambda parameter: 0.001
- Effect: Penalizes large weights, reduces overfitting

#### Dropout
- Implementation with Dropout layers
- Dropout rate: 0.5
- Effect: Random neuron deactivation prevents co-adaptation

### 7. Test Set Evaluation
- Evaluation of best model on test set
- Confusion matrix visualization
- Classification report (precision, recall, F1-score per class)

### 8. Conclusions
- Summary of underfit, good fit, and overfit behaviors
- Analysis of regularization effectiveness
- Discussion of future improvements

## Running the Notebook

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### Execution
1. Navigate to the DL-HW-1 directory
2. Open `scRNA_Complete_Homework_Solution.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially

## Key Results

The notebook demonstrates:
- How model capacity affects fit (underfit vs. good fit vs. overfit)
- How regularization (L2 and Dropout) reduces overfitting
- Proper evaluation methodology (train/val/test split)
- Comprehensive visualization and analysis

## Notes

- All text is in Spanish (as per assignment requirements)
- No Unicode characters are used (as per requirements)
- Code follows best practices for deep learning projects
- Extensive documentation in Markdown cells

## Dependencies

- Python 3.8+
- NumPy
- Pandas  
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow 2.x / Keras

## Author

Deep Learning Course - Homework 1
