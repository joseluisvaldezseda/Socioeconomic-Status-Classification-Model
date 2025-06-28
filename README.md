Socioeconomic Level Classification System (NSE)
Project Description
This project implements a Socioeconomic Level (NSE) classification system using an ensemble of machine learning models. The system can:

✔ Train classification models with historical data
✔ Predict NSE categories for new datasets
✔ Generate insightful visualizations
✔ Evaluate model performance

🗂 Project Structure
text
NSE-Classifier/
│
├── 📜 Building_Classification_Model.py    # Main training/prediction script
├── 📊 Visualizations.py                  # Performance analysis charts
├── 🤖 modelo_nse.pkl                     # Serialized trained model
│
├── 📂 data/                              # Training datasets
│   ├── survey_data_2023.csv
│   └── economic_indicators.xlsx
│
└── 📂 predictions/                       # Prediction outputs
    ├── NSE_Colima_Predictions.xlsx
    └── Jalisco_NSE_Results.csv
⚙️ Requirements
bash
Python 3.7+
pip install -r requirements.txt
Required Packages:

text
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
catboost==1.0.6
xgboost==1.6.1
matplotlib==3.5.3
seaborn==0.11.2
joblib==1.2.0
🚀 Usage
1. Training and Prediction
bash
python Building_Classification_Model.py
What this does:

Trains the ensemble model using data from /data

Saves the trained model as modelo_nse.pkl

Generates predictions for default test data

2. Generating Visualizations
bash
python Visualizations.py
Outputs:

📈 Feature importance plots

🧮 Confusion matrices

📉 Calibration curves

🔍 Error analysis by class

🔗 Feature correlation heatmaps

3. Custom Predictions
Modify in Building_Classification_Model.py:

python
input_path = "data/your_data.csv"   # ← Change this
output_path = "predictions/results.xlsx"  # ← And this
⚙️ Configuration
Variable	Description	Example
data_folder	Training data location	"data/"
nse_combinations	NSE category mapping	{"AB": ["A","B"]}
columns_to_standardize	Features to normalize	["income","education"]
🤖 Ensemble Model
Model	Weight
Random Forest	3
XGBoost	2
CatBoost	2
AdaBoost	1
Extra Trees	1
📊 Results
Output Files:

Excel/CSV files with predictions

Console performance metrics:

Accuracy scores

Classification report

Confusion matrix

Available Visualizations:

Diagram
Code
📝 Notes
⚠️ Important Requirements:

Input files must include all required_columns

Automatically handles missing/infinite values

Excludes invalid classes: IND, ND, C/S, NS

🔮 Future Improvements
🖥️ Graphical user interface

🌐 Prediction API endpoint

🎨 Interactive dashboards

🔧 Hyperparameter optimization

🧩 Extended model persistence with joblib

💡 Tip: For best results, ensure your data follows the same format as the training samples.

diff
+ Ready for production use!
- Under active development
