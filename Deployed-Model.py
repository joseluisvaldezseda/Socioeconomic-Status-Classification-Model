import pandas as pd
import numpy as np
import joblib

# 1. Función para limpiar y convertir a numérico sin perder filas
def limpiar_y_convertir(columna):
    columna_limpia = columna.astype(str).str.replace('*', '0', regex=False).str.strip()
    return pd.to_numeric(columna_limpia, errors='coerce').fillna(0)

# 2. Cargar el modelo entrenado
ruta_modelo = r"C:\Users\jose.valdez\Desktop\python\jose luis\Niveles Socio Economicos\modelo_nse.pkl"
voting_model = joblib.load(ruta_modelo)

# 3. Leer el archivo de predicción
ruta_archivo_parquet = 'Mexico_AGEB_2020.parquet'
df_original = pd.read_parquet(ruta_archivo_parquet, engine='pyarrow')

# Hacemos una copia para trabajar
df_prediccion = df_original.copy()
df_prediccion.columns = df_prediccion.columns.str.strip().str.lower().str.replace('\n', ' ')

# 4. Definir columnas necesarias
columnas_para_estandarizar = [
    'vph_excsa', 'vph_autom', 'vph_inter', 'pocupada', 'pder_ss', 
    'p18ym_pb', 'vph_3ymasc', 'vph_stvp', 'vph_pc', 'vph_cvj', 
    'vph_2ymasd', 'vph_moto', 'vph_bici', 'vph_lavad', 'vph_hmicro', 
    'vph_refri', 'vph_telef', 'vph_spmvpi', 'graproes', 'vph_tv',
    'vph_radio', 'pder_imss', 'vph_1cuart', 'p15sec_co', 'p_0a2', 'p_3a5',
    'p_6a11', 'p_12a14', 'p_15a17', 'p_18a24', 'pob15_64', 
    'p_60ymas'
]

# 5. Procesamiento de denominadores
total_viviendas = limpiar_y_convertir(df_prediccion['tvivparhab']).replace(0, 1)
poblacion_activa = limpiar_y_convertir(df_prediccion['pea']).replace(0, 1)
poblacion_total = limpiar_y_convertir(df_prediccion['pobtot']).replace(0, 1)

# 6. Cálculo de variables (Igual que en el entrenamiento)
calculos = {
    'vph_': total_viviendas,
    'pder_ss': poblacion_total,
    'pder_imss': poblacion_activa,
    'pocupada': poblacion_activa,
    'p18ym_pb': poblacion_activa,
    'p15sec_co': total_viviendas,
    'p_60ymas': poblacion_total
}

for col in columnas_para_estandarizar:
    valor_numerico = limpiar_y_convertir(df_prediccion[col])
    
    if col.startswith('vph_') and col not in ['graproes']:
        df_prediccion[col] = (valor_numerico * 100) / total_viviendas
    elif col in calculos:
        df_prediccion[col] = (valor_numerico * 100) / calculos[col]
    else:
        df_prediccion[col] = valor_numerico

# 7. Limpieza de infinitos
df_prediccion[columnas_para_estandarizar] = df_prediccion[columnas_para_estandarizar].replace([np.inf, -np.inf], np.nan).fillna(0)

# 8. Preparar datos para predicción
X_prediccion = df_prediccion[columnas_para_estandarizar].values.astype(np.float32)

# 9. Realizar las predicciones
print("Realizando predicciones...")
y_prediccion = voting_model.predict(X_prediccion)

# --- NUEVA SECCIÓN: AGREGAR COLUMNAS DE DECISIÓN ---
# Agregamos las columnas procesadas al dataframe original con el prefijo 'calc_'
for col in columnas_para_estandarizar:
    nombre_col_calculada = f"calc_{col}"
    df_original[nombre_col_calculada] = df_prediccion[col].values

# 10. Agregar la predicción final
df_original['nse_predicho'] = y_prediccion

# 11. Opcional: Calcular precisión
if 'nse' in df_original.columns:
    df_eval = df_original.copy()
    df_eval['nse'] = df_eval['nse'].astype(str).str.upper().str.strip()
    validos = ~df_eval['nse'].isin(['IND', 'ND', 'C/S', 'NS', 'NAN', '0', ''])
    df_eval = df_eval[validos]
    
    if len(df_eval) > 0:
        def asignar_clase(nse):
            niveles = ['AB', 'C+', 'C', 'C-', 'D+', 'D', 'E']
            return nse if nse in niveles else None
        df_eval['clase_real'] = df_eval['nse'].apply(asignar_clase)
        correctas = (df_eval['nse_predicho'] == df_eval['clase_real']).sum()
        print(f"Precisión: {(correctas/len(df_eval))*100:.2f}% ({correctas}/{len(df_eval)})")

# 12. Guardar el archivo final
ruta_salida = 'MexicoNSE_Predicciones.parquet'
df_original.to_parquet(ruta_salida, index=False, engine='pyarrow')
print(f"Proceso finalizado. Archivo guardado con columnas de decisión en: {ruta_salida}")

# --- BLOQUE DE RESUMEN FINAL ---
print("\n" + "="*50)
print("RESUMEN ESTADÍSTICO DEL ARCHIVO FINAL")
print("="*50)

# 1. Lista de todas las columnas
print(f"\nLista de columnas ({len(df_original.columns)} en total):")
print(df_original.columns.tolist())

# 2. Shape final (Filas y Columnas)
filas, columnas = df_original.shape
print(f"\nDimensiones finales:")
print(f"Total de Registros (Filas): {filas:,}")
print(f"Total de Variables (Columnas): {columnas:,}")

# 3. Conteo de Predicciones por Clase
print("\nConteo de registros por NSE Predicho:")
conteo_nse = df_original['nse_predicho'].value_counts()
print(conteo_nse)

# 4. Porcentaje de la mezcla de NSE en el archivo
print("\nDistribución porcentual del NSE:")
porcentajes = (df_original['nse_predicho'].value_counts(normalize=True) * 100).round(2)
print(porcentajes.astype(str) + '%')

print("="*50)
