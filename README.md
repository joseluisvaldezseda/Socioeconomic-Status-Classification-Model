# Socioeconomic Level Classification System (NSE)

## Project Description
This project implements a Socioeconomic Level (NSE) classification system using an ensemble model that combines multiple machine learning algorithms. The system can:

- Train a classification model with historical data
- Predict NSE for new datasets
- Generate visualizations to analyze model performance

## Project Structure

Niveles-Socio-Economicos/
│
├── Building Classification Model.py # Main script for training and using the model
├── Visualizations.py # Script for generating analysis charts
├── modelo_nse.pkl # Trained model (generated during execution)
├── data/ # Folder with training data
│ ├── file1.csv
│ ├── file2.csv
│ └── ...
└── predictions/ # Folder for storing results
├── NSE_Colima_NorthAlpha_Predictions.xlsx
└── JaliscoNSE_Predictions.xlsx

## Requirements
- Python 3.7+
- Required libraries (install with `pip install -r requirements.txt`):

pandas
numpy
scikit-learn
catboost
xgboost
matplotlib
seaborn

## Usage

### Training and Prediction
Run the main script:

python "Building Classification Model.py"

This script:

Trains the model with data from the specified folder

Saves the trained model as modelo_nse.pkl

Generates predictions for the Colima dataset

Generating Visualizations
Run the visualization script:
python Visualizations.py


This script generates:

Feature importance charts

Confusion matrix

Calibration curves

Error analysis by class

Feature correlation analysis

Making Predictions on New Data
To predict NSE for new files (e.g., Jalisco), modify the main script to:

Change the input file path

Change the output file path

Configuration
Key variables:

data_folder: Path to folder containing CSV training files

nse_combinations: Dictionary defining NSE groups

columns_to_standardize: List of features used in the model

Ensemble Model
The model combines:

Random Forest (weight: 3)

XGBoost (weight: 2)

CatBoost (weight: 2)

AdaBoost (weight: 1)

Extra Trees (weight: 1)

Results
The model generates:

Excel file with predictions

Performance metrics in console:

Confusion matrix

Classification report

Overall accuracy

Available visualizations:

Feature importance by model

Detailed confusion matrix

Calibration curve

Probability distribution

Classification report (heatmap)

Error analysis by class

Correlation between important features

Notes
Input data must contain all columns specified in required_columns

Null and infinite values are automatically filtered

Invalid classes (IND, ND, C/S, NS) are excluded from analysis

Future Improvements
Implement graphical interface

Create online prediction API

Add more interactive visualizations

Optimize hyperparameters with GridSearch
joblib
