import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder

# 1. Cargar el modelo y los datos necesarios
ruta_modelo = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\modelo_nse.pkl"
voting_model = joblib.load(ruta_modelo)

# Nombres de las características
feature_names = [
    'vph_excsa', 'vph_autom', 'vph_inter', 'pocupada', 'pder_ss', 
    'p18ym_pb', 'vph_3ymasc', 'vph_stvp', 'vph_pc', 'vph_cvj', 
    'vph_2ymasd', 'vph_moto', 'vph_bici', 'vph_lavad', 'vph_hmicro', 
    'vph_refri', 'vph_telef', 'vph_spmvpi', 'graproes', 'vph_tv',
    'vph_radio', 'pder_imss', 'vph_1cuart', 'p15sec_co', 'p_0a2', 'p_3a5',
    'p_6a11', 'p_12a14', 'p_15a17', 'p_18a24', 'pob15_64', 'p_60ymas'
]

# 2. Importancia de características por modelo individual
def plot_individual_feature_importances():
    plt.figure(figsize=(18, 24))
    
    for i, (name, model) in enumerate(voting_model.named_estimators_.items()):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            continue
            
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=False).head(15)
        
        plt.subplot(3, 2, i+1)
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Top 15 Features - {name}')
        plt.tight_layout()
    
    plt.show()

# 3. Matriz de confusión detallada
def plot_detailed_confusion_matrix(X_test, y_test):
    y_pred = voting_model.predict(X_test)
    
    # Codificar las etiquetas si no están numéricas
    if isinstance(y_test[0], str):
        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        y_pred_encoded = le.transform(y_pred)
        classes = le.classes_
    else:
        y_test_encoded = y_test
        y_pred_encoded = y_pred
        classes = voting_model.classes_
    
    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title('Matriz de Confusión Detallada')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

# 4. Curva de calibración (para un modelo de clasificación multiclase)
def plot_calibration_curve(X_test, y_test):
    probas = voting_model.predict_proba(X_test)
    
    # Codificar las etiquetas si no están numéricas
    if isinstance(y_test[0], str):
        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        classes = le.classes_
    else:
        y_test_encoded = y_test
        classes = voting_model.classes_
    
    plt.figure(figsize=(10, 10))
    for i in range(len(classes)):
        true_class = (y_test_encoded == i)
        prob_class = probas[:, i]
        
        prob_true, prob_pred = calibration_curve(true_class, prob_class, n_bins=10)
        
        plt.plot(prob_pred, prob_true, 's-', label=f'{classes[i]}')
    
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectamente calibrado')
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Fracción de clases positivas')
    plt.title('Curva de Calibración')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# 5. Distribución de probabilidades por clase
def plot_probability_distribution(X_test, y_test):
    probas = voting_model.predict_proba(X_test)
    
    if isinstance(y_test[0], str):
        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        classes = le.classes_
    else:
        y_test_encoded = y_test
        classes = voting_model.classes_
    
    plt.figure(figsize=(12, 8))
    for i in range(len(classes)):
        class_probs = probas[y_test_encoded == i, i]
        sns.kdeplot(class_probs, label=f'Clase {classes[i]}', shade=True)
    
    plt.xlabel('Probabilidad asignada a la clase correcta')
    plt.ylabel('Densidad')
    plt.title('Distribución de Probabilidades por Clase')
    plt.legend()
    plt.grid()
    plt.show()

# 6. Reporte de clasificación con mapa de calor
def plot_classification_report(X_test, y_test):
    y_pred = voting_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :3], annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Reporte de Clasificación (Precisión, Recall, F1-score)')
    plt.tight_layout()
    plt.show()

# 7. Gráfico de errores por clase
def plot_class_error_analysis(X_test, y_test):
    y_pred = voting_model.predict(X_test)
    
    if isinstance(y_test[0], str):
        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        y_pred_encoded = le.transform(y_pred)
        classes = le.classes_
    else:
        y_test_encoded = y_test
        y_pred_encoded = y_pred
        classes = voting_model.classes_
    
    error_rates = []
    for i in range(len(classes)):
        total = sum(y_test_encoded == i)
        errors = sum((y_test_encoded == i) & (y_pred_encoded != i))
        error_rates.append(errors / total * 100)
    
    error_df = pd.DataFrame({'Clase': classes, 'Tasa de error (%)': error_rates})
    error_df = error_df.sort_values('Tasa de error (%)', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Tasa de error (%)', y='Clase', data=error_df, palette='Reds_r')
    plt.title('Tasa de Error por Clase')
    plt.xlabel('Tasa de error (%)')
    plt.ylabel('Clase')
    plt.grid(axis='x')
    plt.show()

# 8. Visualización de la correlación entre características importantes
def plot_feature_correlation(X_train, feature_names, top_n=10):
    # Obtener las características más importantes
    total_importance = np.zeros(len(feature_names))
    for name, model in voting_model.named_estimators_.items():
        if hasattr(model, 'feature_importances_'):
            total_importance += model.feature_importances_
    
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': total_importance})
    top_features = importance_df.sort_values('importance', ascending=False).head(top_n)['feature'].values
    
    # Crear DataFrame con las características seleccionadas
    df_top = pd.DataFrame(X_train, columns=feature_names)[top_features]
    
    # Calcular matriz de correlación
    corr = df_top.corr()
    
    # Graficar
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                xticklabels=top_features, yticklabels=top_features)
    plt.title(f'Correlación entre las {top_n} Características más Importantes')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Ejemplo de cómo usar estas funciones (necesitarás X_test, y_test, X_train)
if __name__ == "__main__":
    # Necesitas cargar tus datos de prueba y entrenamiento si quieres usar todas las visualizaciones
    # Por ejemplo:
    # Ruta de la carpeta que contiene los archivos CSV
    carpeta_datos = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\NSE"

    combinaciones_nse = {
        'AB': ['AB'],
        'C+': ['C+'],
        'C': ['C'],
        'C-': ['C-'],
        'D+': ['D+'],
        'D': ['D'],
        'E':['E']
    }

    # Función para asignar clase socioeconómica
    def asignar_clase(nse):
        for clase, niveles in combinaciones_nse.items():
            if nse in niveles:
                return clase
        return None

    # Función para limpiar y convertir columnas a numérico, eliminando asteriscos
    def limpiar_y_convertir(columna):
        columna_str = columna.astype(str)
        columna_limpia = columna_str.str.replace('*', '', regex=False)
        return pd.to_numeric(columna_limpia, errors='coerce')

    # Inicializar listas para los datos de entrenamiento
    X_lista = []
    y_lista = []

    # Leer y procesar cada archivo CSV en la carpeta
    for archivo in os.listdir(carpeta_datos):
        if archivo.endswith(".csv"):
            ruta_archivo_csv = os.path.join(carpeta_datos, archivo)
            df = pd.read_csv(ruta_archivo_csv)
            
            # Limpiar nombres de columnas y ajustar
            df.columns = df.columns.str.strip().str.lower().str.replace('\n', ' ')

            # Universo:
            total_viviendas = limpiar_y_convertir(df['tvivparhab'])  # Total de viviendas particulares habitadas
            poblacion_activa = limpiar_y_convertir(df['pea'])
            poblacion_total = limpiar_y_convertir(df['pobtot'])

            # Cálculo de las variables
            df['vph_excsa'] = limpiar_y_convertir(df['vph_excsa']) * 100 / total_viviendas
            df['vph_autom'] = limpiar_y_convertir(df['vph_autom']) * 100 / total_viviendas
            df['vph_inter'] = limpiar_y_convertir(df['vph_inter']) * 100 / total_viviendas
            df['pocupada'] = limpiar_y_convertir(df['pocupada']) * 100 / poblacion_activa  # Población Económicamente Activa
            df['pder_ss'] = limpiar_y_convertir(df['pder_ss']) * 100 / poblacion_total 
            df['p18ym_pb'] = limpiar_y_convertir(df['p18ym_pb']) * 100 / poblacion_activa
            df['vph_3ymasc'] = limpiar_y_convertir(df['vph_3ymasc']) * 100 / total_viviendas
            df['vph_stvp'] = limpiar_y_convertir(df['vph_stvp']) * 100 / total_viviendas
            df['vph_pc'] = limpiar_y_convertir(df['vph_pc']) * 100 / total_viviendas
            df['vph_cvj'] = limpiar_y_convertir(df['vph_cvj']) * 100 / total_viviendas
            df['vph_2ymasd'] = limpiar_y_convertir(df['vph_2ymasd']) * 100 / total_viviendas
            df['vph_moto'] = limpiar_y_convertir(df['vph_moto']) * 100 / total_viviendas
            df['vph_bici'] = limpiar_y_convertir(df['vph_bici']) * 100 / total_viviendas
            df['vph_lavad'] = limpiar_y_convertir(df['vph_lavad']) * 100 / total_viviendas
            df['vph_hmicro'] = limpiar_y_convertir(df['vph_hmicro']) * 100 / total_viviendas
            df['vph_refri'] = limpiar_y_convertir(df['vph_refri']) * 100 / total_viviendas
            df['vph_telef'] = limpiar_y_convertir(df['vph_telef']) * 100 / total_viviendas
            df['vph_spmvpi'] = limpiar_y_convertir(df['vph_spmvpi']) * 100 / total_viviendas
            df['vph_tv'] = limpiar_y_convertir(df['vph_tv']) * 100 / total_viviendas
            df['vph_radio'] = limpiar_y_convertir(df['vph_radio']) * 100 / total_viviendas
            df['pder_imss'] = limpiar_y_convertir(df['pder_imss']) * 100 / poblacion_activa 
            df['vph_1cuart'] = limpiar_y_convertir(df['vph_1cuart']) * 100 / total_viviendas
            df['p15sec_co'] = limpiar_y_convertir(df['p15sec_co']) * 100 / total_viviendas
            df['p_60ymas'] = limpiar_y_convertir(df['p_60ymas']) * 100 / poblacion_total 

            # Filtrar filas con valores nulos
            columnas_a_filtrar =  [
                                    'vph_excsa', 'vph_autom', 'vph_inter', 'pocupada', 'pder_ss', 
                                    'p18ym_pb', 'vph_3ymasc', 'vph_stvp', 'vph_pc', 'vph_cvj', 
                                    'vph_2ymasd', 'vph_moto', 'vph_bici', 'vph_lavad', 'vph_hmicro', 
                                    'vph_refri', 'vph_telef', 'vph_spmvpi', 'graproes' ,'vph_tv',
                                    'vph_radio','pder_imss','vph_1cuart','p15sec_co', 'p_60ymas'
                                ]

            df = df.dropna(subset=columnas_a_filtrar)
            df = df[~df[columnas_a_filtrar].isin([np.inf, -np.inf]).any(axis=1)]
            df['nse'] = df['nse'].astype('category')
            df = df[~df['nse'].isin(['IND', 'ND', 'C/S','NS'])]
            
            # Seleccionar columnas de análisis
            columnas_para_estandarizar = [
                                            'vph_excsa', 'vph_autom', 'vph_inter', 'pocupada', 'pder_ss', 
                                            'p18ym_pb', 'vph_3ymasc', 'vph_stvp', 'vph_pc', 'vph_cvj', 
                                            'vph_2ymasd', 'vph_moto', 'vph_bici', 'vph_lavad', 'vph_hmicro', 
                                            'vph_refri', 'vph_telef', 'vph_spmvpi', 'graproes','vph_tv',
                                            'vph_radio','pder_imss','vph_1cuart','p15sec_co','p_0a2','p_3a5',
        'p_6a11','p_12a14','p_15a17','p_18a24','pob15_64','p_60ymas'
                                        ]

            # Aplicar LabelEncoder a la columna 'nse'
            #df['nse'] = label_encoder.fit_transform(df['nse'])
            # Agregar datos a la lista de entrenamiento
            #
            df['clase'] = df['nse'].apply(asignar_clase)
            df['clase'] = df['clase'].astype('category')
            X_lista.append(df[columnas_para_estandarizar].values)
            y_lista.append(df['clase'].values)

    # Concatenar los datos para entrenar el modelo final
    X_total = np.vstack(X_lista)
    y_total = np.concatenate(y_lista)

    # Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, random_state=42)
    
    # Visualizaciones que no requieren datos de entrada
    plot_individual_feature_importances()
    
    # Visualizaciones que requieren datos de prueba
    plot_detailed_confusion_matrix(X_test, y_test)
    plot_calibration_curve(X_test, y_test)
    plot_probability_distribution(X_test, y_test)
    plot_classification_report(X_test, y_test)
    plot_class_error_analysis(X_test, y_test)
    plot_feature_correlation(X_train, feature_names)
