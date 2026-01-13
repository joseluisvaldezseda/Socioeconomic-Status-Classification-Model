# Socioeconomic Status (SES) Classifier for Mexico

Machine Learning ensemble model that classifies geographic areas according to their socioeconomic level using census data from INEGI 2020 (https://www.inegi.org.mx/programas/ccpv/2020/#datos_abiertos).

## 🌐 Live Demo

Try the model online: [https://socioeconomic-status-classification-model.streamlit.app/](https://socioeconomic-status-classification-model.streamlit.app/)

## 📊 Project Description

This project implements an SES classification system for Mexico using demographic and socioeconomic variables from the census. The model predicts 7 socioeconomic categories: **AB, C+, C, C-, D+, D, and E**.

## 🎯 Model Performance

### Global Metrics
- **Test Accuracy**: 71.61%
- **Real Prediction Accuracy**: 83.87%
- **Total samples evaluated**: 107,887

### Confusion Matrix
```
              AB      C      C+     C-      D      D+      E
AB         5,291    111   2,117     7      0      2      0
C             47 10,037  1,981  2,970    128    677      0
C+         1,200  2,379  8,893    200     11     32      0
C-            38  2,941    246 23,549    886  3,686     10
D              0    102      2    957 13,386  2,920     67
D+             3    665     21  2,431  3,326 15,927      3
E              0      0      0    109    348      1    180
```

### Metrics by Class

| Class | Precision | Recall | F1-Score | Support | Notes |
|-------|-----------|--------|----------|---------|-------|
| AB    | 0.80      | 0.70   | 0.75     | 7,528   | High socioeconomic level |
| C     | 0.62      | 0.63   | 0.63     | 15,840  | Confusion with C+ and C- |
| C+    | 0.67      | 0.70   | 0.68     | 12,715  | Good separation |
| C-    | 0.78      | 0.75   | 0.76     | 31,356  | Best performance |
| D     | 0.74      | 0.77   | 0.75     | 17,434  | Good balance |
| D+    | 0.69      | 0.71   | 0.70     | 22,376  | Confusion with D and C- |
| E     | 0.69      | 0.28   | 0.40     | 638     | Minority class |

**Weighted average**: Precision 0.72, Recall 0.72, F1-Score 0.72

## 🧠 Model Architecture

### Voting Classifier (Soft Voting)
Ensemble of 5 classifiers with optimized weights:

1. **Random Forest** (weight: 3)
   - n_estimators: 200
   - max_depth: 20
   - min_samples_split: 2
   - min_samples_leaf: 2
   
2. **XGBoost** (weight: 2)
   - objective: 'multi:softmax'
   - n_estimators: 200
   - max_depth: 10
   - learning_rate: 0.1
   
3. **CatBoost** (weight: 2)
   - iterations: 300
   - learning_rate: 0.1
   - depth: 10
   
4. **AdaBoost** (weight: 1)
   - n_estimators: 200
   
5. **Extra Trees** (weight: 1)
   - n_estimators: 200
   - max_depth: 20

### Validation Strategy
- Split: 80% training / 20% test
- Random state: 42
- No standardization (improves performance with tree-based models)

## 📝 Features Used (32 features)

### Housing Variables (% of total households)
- `vph_excsa`: Households with toilet
- `vph_autom`: Households with automobile
- `vph_inter`: Households with internet
- `vph_3ymasc`: Households with 3 or more rooms
- `vph_stvp`: Households without pay TV
- `vph_pc`: Households with computer
- `vph_cvj`: Households with video game console
- `vph_2ymasd`: Households with 2 or more bedrooms
- `vph_moto`: Households with motorcycle
- `vph_bici`: Households with bicycle
- `vph_lavad`: Households with washing machine
- `vph_hmicro`: Households with microwave oven
- `vph_refri`: Households with refrigerator
- `vph_telef`: Households with telephone
- `vph_spmvpi`: Households without any goods
- `vph_tv`: Households with television
- `vph_radio`: Households with radio
- `vph_1cuart`: Households with 1 room

### Economically Active Population Variables
- `pocupada`: Employed population (% of EAP)
- `p18ym_pb`: Population 18 years and older with basic primary education (% of EAP)
- `pder_imss`: Population entitled to IMSS (% of EAP)

### Total Population Variables
- `pder_ss`: Population with right to health services (% of total population)
- `p15sec_co`: Population 15 years and older with complete secondary education
- `p_60ymas`: Population 60 years and older

### Demographic Variables by Age
- `p_0a2`: Population 0 to 2 years
- `p_3a5`: Population 3 to 5 years
- `p_6a11`: Population 6 to 11 years
- `p_12a14`: Population 12 to 14 years
- `p_15a17`: Population 15 to 17 years
- `p_18a24`: Population 18 to 24 years
- `pob15_64`: Population 15 to 64 years

### Other Variables
- `graproes`: Average years of schooling

## 🚀 Usage

### 1. Model Training

```python
python Building_Classification_Model.py
```

**Required input**:
- Folder with training CSV files (INEGI data)
- Each file must include `nse` column with actual classification
- Required demographic and housing variables

**Output**:
- Trained serialized model
- Classification report in console
- Excel file with predictions

### 2. Prediction with Trained Model

```python
python Deployed-Model.py
```

**Required input**:
- Pre-trained model
- CSV/Excel file with data to predict

**Output**:
- Excel file with `nse_predicho` column
- Accuracy metrics (if actual `nse` column exists)

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

### Recommended versions
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
catboost>=1.2
xgboost>=1.7.0
joblib>=1.2.0
openpyxl>=3.0.0
streamlit>=1.28.0
```

## 📂 Project Structure

```
Socioeconomic-Status-Classification-Model/
│
├── app.py                              # Streamlit web application
├── Building_Classification_Model.py    # Model training script
├── Deployed-Model.py                   # Prediction script
├── Model_Performance.py                # Performance visualization script
├── Preprocessing_ENIGH_Data.py         # Data preprocessing script
├── README.md                           # This file
├── requirements.txt                    # Project dependencies
│
├── Comparison.png                      # Model comparison visualization
├── Glosario_Variables.pdf              # Variable glossary
├── MexicoNSE_Predicciones.parquet     # Predictions dataset
│
└── data/                               # Data folder (not included in repo)
    ├── training/                       # Training data
    │   ├── state1.csv
    │   ├── state2.csv
    │   └── ...
    └── prediction/
        └── new_data.csv                # Data to classify
```

## 🔧 Data Preprocessing

### Cleaning
- Removal of asterisks (*) in numeric values
- Conversion to numeric type with error handling
- Filtering of null and infinite values

### Transformations
All housing and population variables are expressed as percentages:
- Housing variables: `(value / total_households) * 100`
- EAP variables: `(value / active_population) * 100`
- Demographic variables: `(value / total_population) * 100`

### Category Filtering
The following categories are excluded:
- `IND`: Undetermined
- `ND`: Not available
- `C/S`: With/Without data
- `NS`: Not specified

## 📊 Results Interpretation

### SES Categories
- **AB**: High socioeconomic level
- **C+**: Upper-middle level
- **C**: Middle level
- **C-**: Lower-middle level
- **D+**: Upper-low level
- **D**: Low level
- **E**: Very low level

### Model Observations
1. **Best performance**: C- and AB classes (F1-Score > 0.75)
2. **Main challenge**: Class E due to data imbalance (only 638 samples)
3. **Common confusion**: Between adjacent levels (C, C+, C-)
4. **Strength**: High accuracy at extremes (AB and D)

## 🎓 Applications

- **Market analysis**: Geographic segmentation for commercial strategies
- **Public policy**: Identification of priority areas for social programs
- **Social research**: Demographic and socioeconomic studies
- **Urban planning**: Infrastructure and service development

## ⚠️ Limitations

1. **Class imbalance**: Class E is underrepresented
2. **Confusion between middle levels**: C, C+, and C- have similar characteristics
3. **Census data dependency**: Requires periodic updates
4. **Geographic context**: Trained with Mexican data (INEGI)

## 🔄 Future Improvements

- [ ] Implement oversampling techniques for class E (SMOTE)
- [ ] Add stratified cross-validation
- [ ] Include geospatial variables (latitude/longitude)
- [ ] Experiment with neural networks to capture complex interactions
- [ ] Create REST API for real-time predictions
- [ ] Implement model explainability (SHAP values)

## 👥 Author

**José Luis Valdez**

## 📄 License

This project uses public data from INEGI (National Institute of Statistics and Geography of Mexico).

## 🙏 Acknowledgments

- INEGI for providing high-quality census data
- scikit-learn, XGBoost, and CatBoost communities for their excellent tools
- Streamlit for enabling easy deployment of ML applications

---

**Last updated**: January 2026  
**Model version**: 1.0  
**Production accuracy**: 83.87%

