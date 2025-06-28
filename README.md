Sistema de Clasificación Socioeconómica (NSE)
Descripción del Proyecto
Este proyecto implementa un sistema de clasificación de Niveles Socioeconómicos (NSE) utilizando un modelo de ensamble que combina múltiples algoritmos de machine learning. El sistema puede:

Entrenar un modelo de clasificación con datos históricos

Predecir el NSE para nuevos conjuntos de datos

Generar visualizaciones para analizar el rendimiento del modelo

Estructura del Proyecto
text
Niveles-Socio-Economicos/
│
├── Building Classification Model.py  # Script principal para entrenar y usar el modelo
├── Visualizations.py                # Script para generar gráficos de análisis
├── modelo_nse.pkl                   # Modelo entrenado (generado al ejecutar)
├── data/                            # Carpeta con datos de entrenamiento
│   ├── archivo1.csv
│   ├── archivo2.csv
│   └── ...
└── predicciones/                    # Carpeta para guardar resultados
    ├── NSE_Colima_NorthAlpha_Predicciones.xlsx
    └── JaliscoNSE_Predicciones.xlsx
Requisitos
Python 3.7+

Bibliotecas requeridas (instalar con pip install -r requirements.txt):

text
pandas
numpy
scikit-learn
catboost
xgboost
matplotlib
seaborn
joblib
Uso
1. Entrenamiento y predicción
Ejecutar el script principal:

bash
python "Building Classification Model.py"
Este script:

Entrena el modelo con los datos en la carpeta especificada

Guarda el modelo entrenado en modelo_nse.pkl

Genera predicciones para el archivo de Colima

2. Generar visualizaciones
Ejecutar el script de visualización:

bash
python Visualizations.py
Este script genera:

Gráficos de importancia de características

Matriz de confusión

Curvas de calibración

Análisis de errores por clase

Correlación entre características

3. Hacer predicciones para nuevos datos
Para predecir NSE en nuevos archivos (ej. Jalisco), modificar el script principal para:

Cambiar la ruta del archivo de entrada

Cambiar la ruta del archivo de salida

Configuración
Variables importantes
carpeta_datos: Ruta a la carpeta con archivos CSV de entrenamiento

combinaciones_nse: Diccionario que define los grupos de NSE

columnas_para_estandarizar: Lista de características usadas en el modelo

Modelo de ensamble
El modelo combina:

Random Forest

XGBoost

CatBoost

AdaBoost

Extra Trees

Con pesos: [3, 2, 2, 1, 1] respectivamente

Resultados
El modelo genera:

Archivo Excel con las predicciones

Métricas de rendimiento en consola:

Matriz de confusión

Reporte de clasificación

Precisión global

Visualizaciones disponibles
Importancia de características por modelo

Matriz de confusión detallada

Curva de calibración

Distribución de probabilidades

Reporte de clasificación (heatmap)

Análisis de errores por clase

Correlación entre características importantes

Notas
Los datos de entrada deben contener todas las columnas especificadas en columnas_requeridas

Se filtran automáticamente valores nulos e infinitos

Las clases no válidas (IND, ND, C/S, NS) se excluyen del análisis

Mejoras futuras
Implementar interfaz gráfica

Crear API para predicciones en línea

Añadir más visualizaciones interactivas

Optimizar hiperparámetros con GridSearch

