import pandas as pd
import zipfile
import os
import glob

# 1. Configuración de rutas
ruta_busqueda = r'C:\Users\jose.valdez\Downloads\conjunto_de_datos\info'
ruta_salida = r'C:\Users\jose.valdez\Downloads\conjunto_de_datos\Mexico_AGEB_2020.parquet'

lista_mexico = []

print("Iniciando la consolidación de 32 estados con manejo de codificación...")

for i in range(1, 33):
    entidad = str(i).zfill(2)
    patron = os.path.join(ruta_busqueda, f'RESAGEBURB_{entidad}_*.zip')
    archivos_encontrados = glob.glob(patron)
    
    if not archivos_encontrados:
        continue
        
    archivo_zip = archivos_encontrados[0]
    
    try:
        with zipfile.ZipFile(archivo_zip, 'r') as z:
            archivos_internos = [f for f in z.namelist() if f.lower().endswith('.csv')]
            
            for nombre_csv in archivos_internos:
                with z.open(nombre_csv) as f:
                    # Intentamos leer primero con latin-1 que es común en archivos de México
                    try:
                        df_estado = pd.read_csv(f, low_memory=False, encoding='latin-1')
                    except UnicodeDecodeError:
                        # Si falla, intentamos con utf-8
                        f.seek(0) # Regresar al inicio del archivo interno
                        df_estado = pd.read_csv(f, low_memory=False, encoding='utf-8')
                    
                    # Estandarizamos nombres de columnas
                    df_estado.columns = df_estado.columns.str.lower()
                    
                    # FILTRO: Solo totales por AGEB (Manzana 0 o 000)
                    if 'mza' in df_estado.columns:
                        df_estado['mza'] = df_estado['mza'].astype(str)
                        df_estado = df_estado[df_estado['mza'].isin(['0', '000'])]
                    
                    lista_mexico.append(df_estado)
                    print(f"✅ Procesado: Entidad {entidad}")
                    
    except Exception as e:
        print(f"❌ Error crítico en {archivo_zip}: {e}")

# 3. Concatenar y guardar
if lista_mexico:
    print("\nConcatenando todos los estados... esto puede tardar un poco.")
    df_completo = pd.concat(lista_mexico, ignore_index=True)
    
    # Asegurar que todas las columnas sean strings o números antes de guardar en Parquet
    # Parquet a veces se queja si hay columnas con tipos mezclados (objetos)
    for col in df_completo.columns:
        if df_completo[col].dtype == 'object':
            df_completo[col] = df_completo[col].astype(str)

    df_completo.to_parquet(ruta_salida, index=False, engine='pyarrow')
    
    print("-" * 30)
    print(f"¡ÉXITO! Archivo México_AGEB_2020.parquet creado.")
    print(f"Total de filas: {len(df_completo)}")
    print("-" * 30)
