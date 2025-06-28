
# Socioeconomic Level Classification System (NSE)

## Project Description  
This project implements a **Socioeconomic Level (NSE) classification system** using machine learning. Key capabilities:

- Train classification models with historical data
- Predict NSE categories for new datasets
- Generate performance visualizations
- Export prediction results

## Project Structure  
Niveles-Socio-Economicos/
│
├── Building_Classification_Model.py # Main training/prediction script
├── Visualizations.py # Analysis charts generator
├── modelo_nse.pkl # Serialized trained model
│
├── data/ # Training datasets
│ ├── survey_data.csv
│ └── economic_indicators.csv
│
└── predictions/ # Output folder
├── NSE_Colima_Results.xlsx
└── Jalisco_Predictions.csv

text

## Requirements  
Python 3.7+
pip install -r requirements.txt
Required packages:

text
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
catboost>=1.0.0
xgboost>=1.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.2.0
Usage
Training and Prediction
bash
python Building_Classification_Model.py
Generating Visualizations
bash
python Visualizations.py
Custom Predictions
Modify these paths in the script:

python
input_path = "data/your_data.csv"
output_path = "predictions/your_results.xlsx"
Model Configuration
Ensemble Weights:

Random Forest: 3

XGBoost: 2

CatBoost: 2

AdaBoost: 1

Extra Trees: 1

Key Variables:

data_folder: Path to training data

nse_combinations: Category mapping dictionary

columns_to_standardize: Features to normalize

Outputs
Generated Files:

Excel/CSV prediction files

Model performance metrics

Visualization charts

Notes
Input files must contain all required columns

Automatically handles missing values

Excludes invalid classes: IND, ND, C/S, NS

Future Improvements
Graphical interface

Web API endpoint

Interactive dashboards

Advanced hyperparameter tuning
