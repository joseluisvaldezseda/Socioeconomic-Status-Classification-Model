import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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


# Estandarización
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)  # Ajustar y transformar el entrenamiento
#X_test = scaler.transform(X_test)       # Transformar el conjunto de prueba


 
# Definir los modelos individuales
random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42
)

xgb_model = XGBClassifier(
    objective='multi:softmax',
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)

catboost_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=10,
    verbose=0,  # Silenciar la salida durante el entrenamiento
    random_state=42
)


#logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
#svm_model = SVC(kernel='linear', probability=True, random_state=42)  # 'probability=True' permite la votación soft
#knn_model = KNeighborsClassifier(n_neighbors=5)
#gradient_boosting = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
#lightgbm_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)
adaboost_model = AdaBoostClassifier(n_estimators=200, random_state=42)
extra_trees = ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42)


# Crear el modelo de votación
voting_model = VotingClassifier(
    estimators=[
        ('rf', random_forest),
        ('xgb', xgb_model),
        ('catboost', catboost_model),
        ('adaboost', adaboost_model),
        ('extra_trees', extra_trees),
    ],
    voting='soft',  # Cambia a 'hard' si prefieres la votación mayoritaria,
    weights=[3, 2, 2, 1, 1]  # Ajustar según rendimiento individual
)

# Entrenar el modelo de votación
voting_model.fit(X_train, y_train)

# Evaluar el modelo 
y_pred = voting_model.predict(X_test)

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nPrecisión del modelo:")
print(accuracy_score(y_test, y_pred))

# Ruta del archivo de entrada (excel con datos a predecir)
ruta_archivo_excel = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\NSE Colima NorthAlpha.xlsx"

# Leer el archivo de predicción
df_prediccion = pd.read_excel(ruta_archivo_excel)

# Limpiar nombres de columnas y ajustar (convertir a minúsculas y eliminar espacios/nuevas líneas)
df_prediccion.columns = df_prediccion.columns.str.strip().str.lower().str.replace('\n', ' ')

# Asegurarse de que las columnas coincidan exactamente con las usadas en el entrenamiento
columnas_requeridas = [
    'tvivparhab', 'pea', 'pobtot', 'vph_excsa', 'vph_autom', 'vph_inter', 
    'pocupada', 'pder_ss', 'p18ym_pb', 'vph_3ymasc', 'vph_stvp', 'vph_pc', 
    'vph_cvj', 'vph_2ymasd', 'vph_moto', 'vph_bici', 'vph_lavad', 'vph_hmicro', 
    'vph_refri', 'vph_telef', 'vph_spmvpi', 'graproes', 'vph_tv', 'vph_radio',
    'pder_imss', 'vph_1cuart', 'p15sec_co', 'p_60ymas'
]

# Verificar que todas las columnas requeridas estén presentes
columnas_faltantes = [col for col in columnas_requeridas if col not in df_prediccion.columns]
if columnas_faltantes:
    raise ValueError(f"Columnas faltantes en el archivo de predicción: {columnas_faltantes}")

# Procesamiento de datos igual que en el entrenamiento
total_viviendas = limpiar_y_convertir(df_prediccion['tvivparhab'])
poblacion_activa = limpiar_y_convertir(df_prediccion['pea'])
poblacion_total = limpiar_y_convertir(df_prediccion['pobtot'])

# Cálculo de las variables (igual que en el entrenamiento)
df_prediccion['vph_excsa'] = limpiar_y_convertir(df_prediccion['vph_excsa']) * 100 / total_viviendas
df_prediccion['vph_autom'] = limpiar_y_convertir(df_prediccion['vph_autom']) * 100 / total_viviendas
df_prediccion['vph_inter'] = limpiar_y_convertir(df_prediccion['vph_inter']) * 100 / total_viviendas
df_prediccion['pocupada'] = limpiar_y_convertir(df_prediccion['pocupada']) * 100 / poblacion_activa
df_prediccion['pder_ss'] = limpiar_y_convertir(df_prediccion['pder_ss']) * 100 / poblacion_total
df_prediccion['p18ym_pb'] = limpiar_y_convertir(df_prediccion['p18ym_pb']) * 100 / poblacion_activa
df_prediccion['vph_3ymasc'] = limpiar_y_convertir(df_prediccion['vph_3ymasc']) * 100 / total_viviendas
df_prediccion['vph_stvp'] = limpiar_y_convertir(df_prediccion['vph_stvp']) * 100 / total_viviendas
df_prediccion['vph_pc'] = limpiar_y_convertir(df_prediccion['vph_pc']) * 100 / total_viviendas
df_prediccion['vph_cvj'] = limpiar_y_convertir(df_prediccion['vph_cvj']) * 100 / total_viviendas
df_prediccion['vph_2ymasd'] = limpiar_y_convertir(df_prediccion['vph_2ymasd']) * 100 / total_viviendas
df_prediccion['vph_moto'] = limpiar_y_convertir(df_prediccion['vph_moto']) * 100 / total_viviendas
df_prediccion['vph_bici'] = limpiar_y_convertir(df_prediccion['vph_bici']) * 100 / total_viviendas
df_prediccion['vph_lavad'] = limpiar_y_convertir(df_prediccion['vph_lavad']) * 100 / total_viviendas
df_prediccion['vph_hmicro'] = limpiar_y_convertir(df_prediccion['vph_hmicro']) * 100 / total_viviendas
df_prediccion['vph_refri'] = limpiar_y_convertir(df_prediccion['vph_refri']) * 100 / total_viviendas
df_prediccion['vph_telef'] = limpiar_y_convertir(df_prediccion['vph_telef']) * 100 / total_viviendas
df_prediccion['vph_spmvpi'] = limpiar_y_convertir(df_prediccion['vph_spmvpi']) * 100 / total_viviendas
df_prediccion['graproes'] = limpiar_y_convertir(df_prediccion['graproes'])
df_prediccion['vph_tv'] = limpiar_y_convertir(df_prediccion['vph_tv']) * 100 / total_viviendas
df_prediccion['vph_radio'] = limpiar_y_convertir(df_prediccion['vph_radio']) * 100 / total_viviendas
df_prediccion['pder_imss'] = limpiar_y_convertir(df_prediccion['pder_imss']) * 100 / poblacion_activa
df_prediccion['vph_1cuart'] = limpiar_y_convertir(df_prediccion['vph_1cuart']) * 100 / total_viviendas
df_prediccion['p15sec_co'] = limpiar_y_convertir(df_prediccion['p15sec_co']) * 100 / total_viviendas
df_prediccion['p_60ymas'] = limpiar_y_convertir(df_prediccion['p_60ymas']) * 100 / poblacion_total

# Seleccionar exactamente las mismas columnas usadas en el entrenamiento
# (debe tener 32 columnas, igual que en el entrenamiento)
columnas_para_estandarizar = [
    'vph_excsa', 'vph_autom', 'vph_inter', 'pocupada', 'pder_ss', 
    'p18ym_pb', 'vph_3ymasc', 'vph_stvp', 'vph_pc', 'vph_cvj', 
    'vph_2ymasd', 'vph_moto', 'vph_bici', 'vph_lavad', 'vph_hmicro', 
    'vph_refri', 'vph_telef', 'vph_spmvpi', 'graproes', 'vph_tv',
    'vph_radio', 'pder_imss', 'vph_1cuart', 'p15sec_co', 'p_0a2', 'p_3a5',
    'p_6a11', 'p_12a14', 'p_15a17', 'p_18a24', 'pob15_64', 'p_60ymas'
]

# Filtrar filas con valores nulos
df_prediccion = df_prediccion.dropna(subset=columnas_para_estandarizar)
df_prediccion = df_prediccion[~df_prediccion[columnas_para_estandarizar].isin([np.inf, -np.inf]).any(axis=1)]

# Preparar datos para predicción
X_prediccion = df_prediccion[columnas_para_estandarizar].values

# Realizar las predicciones
y_prediccion = voting_model.predict(X_prediccion)

# Agregar las predicciones al DataFrame
df_prediccion['nse_predicho'] = y_prediccion

# Si existe columna 'nse' en los datos de predicción, calcular precisión
if 'nse' in df_prediccion.columns:
    df_prediccion['nse'] = df_prediccion['nse'].astype(str).str.upper()
    df_prediccion = df_prediccion[~df_prediccion['nse'].isin(['IND', 'ND', 'C/S', 'NS'])]
    
    # Aplicar la misma función de asignación de clase
    df_prediccion['clase_real'] = df_prediccion['nse'].apply(asignar_clase)
    
    # Calcular precisión solo si hay valores reales para comparar
    if 'clase_real' in df_prediccion.columns:
        predicciones_correctas = (df_prediccion['nse_predicho'] == df_prediccion['clase_real']).sum()
        total_predicciones = len(df_prediccion)
        porcentaje_correcto = (predicciones_correctas / total_predicciones) * 100
        print(f"\nPorcentaje de predicciones correctas: {porcentaje_correcto:.2f}%")

# Guardar el modelo entrenado
import joblib
ruta_modelo = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\modelo_nse.pkl"
joblib.dump(voting_model, ruta_modelo)
print(f"Modelo guardado en {ruta_modelo}")

# Guardar las predicciones
ruta_salida = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\NSE_Colima_NorthAlpha_Predicciones.xlsx"
df_prediccion.to_excel(ruta_salida, index=False)
print(f"Predicciones guardadas en {ruta_salida}")