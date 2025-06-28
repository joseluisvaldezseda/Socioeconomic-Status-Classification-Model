import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Función para limpiar y convertir columnas a numérico, eliminando asteriscos
def limpiar_y_convertir(columna):
    columna_str = columna.astype(str)
    columna_limpia = columna_str.str.replace('*', '', regex=False)
    return pd.to_numeric(columna_limpia, errors='coerce')

# Cargar el modelo entrenado
ruta_modelo = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\modelo_nse.pkl"
voting_model = joblib.load(ruta_modelo)

# Ruta del archivo de entrada (CSV con datos a predecir)
ruta_archivo_csv = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\José Luis\JaliscoNSE.csv"

# Leer el archivo de predicción
df_prediccion = pd.read_csv(ruta_archivo_csv)

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
    
    # Función para asignar clase socioeconómica (la misma que usaste en el entrenamiento)
    def asignar_clase(nse):
        combinaciones_nse = {
            'AB': ['AB'],
            'C+': ['C+'],
            'C': ['C'],
            'C-': ['C-'],
            'D+': ['D+'],
            'D': ['D'],
            'E': ['E']
        }
        for clase, niveles in combinaciones_nse.items():
            if nse in niveles:
                return clase
        return None
    
    # Aplicar la misma función de asignación de clase
    df_prediccion['clase_real'] = df_prediccion['nse'].apply(asignar_clase)
    
    # Calcular precisión solo si hay valores reales para comparar
    if 'clase_real' in df_prediccion.columns:
        predicciones_correctas = (df_prediccion['nse_predicho'] == df_prediccion['clase_real']).sum()
        total_predicciones = len(df_prediccion)
        porcentaje_correcto = (predicciones_correctas / total_predicciones) * 100
        print(f"\nPorcentaje de predicciones correctas: {porcentaje_correcto:.2f}%")

# Guardar las predicciones
ruta_salida = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\JaliscoNSE_Predicciones.xlsx"
df_prediccion.to_excel(ruta_salida, index=False)
print(f"Predicciones guardadas en {ruta_salida}")
