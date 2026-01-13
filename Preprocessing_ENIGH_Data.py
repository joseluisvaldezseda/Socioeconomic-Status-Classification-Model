import pandas as pd
import zipfile
import os
import glob

# 1. Configuración de rutas
ruta_busqueda = r'C:\Users\jose.valdez\Downloads\conjunto_de_datos\info'
ruta_salida = r'C:\Users\jose.valdez\Downloads\conjunto_de_datos\Mexico_Completo_2020.parquet'

lista_mexico = []

print("Iniciando la unión masiva de archivos (todas las manzanas de México)...")

for i in range(1, 33):
    entidad = str(i).zfill(2)
    patron = os.path.join(ruta_busqueda, f'RESAGEBURB_{entidad}_*.zip')
    archivos_encontrados = glob.glob(patron)
    
    if not archivos_encontrados:
        print(f"⚠️ No se encontró el archivo para la entidad {entidad}")
        continue
        
    archivo_zip = archivos_encontrados[0]
    
    try:
        with zipfile.ZipFile(archivo_zip, 'r') as z:
            # Buscamos archivos CSV dentro del ZIP
            archivos_internos = [f for f in z.namelist() if f.lower().endswith('.csv')]
            
            for nombre_csv in archivos_internos:
                with z.open(nombre_csv) as f:
                    # Leemos SIN filtros, cargando todo tal cual viene
                    try:
                        # Usamos latin-1 y low_memory=False para cargar todo sin errores
                        df_estado = pd.read_csv(f, low_memory=False, encoding='latin-1')
                    except UnicodeDecodeError:
                        f.seek(0)
                        df_estado = pd.read_csv(f, low_memory=False, encoding='utf-8')
                    
                    # Convertimos columnas a minúsculas solo para asegurar que se apilen correctamente
                    df_estado.columns = df_estado.columns.str.lower()
                    
                    lista_mexico.append(df_estado)
                    print(f"✅ Apilado: Entidad {entidad} - Filas añadidas: {len(df_estado)}")
                    
    except Exception as e:
        print(f"❌ Error en {archivo_zip}: {e}")

# 3. Concatenación final
if lista_mexico:
    print("\nConcatenando todos los estados en un solo bloque...")
    df_completo = pd.concat(lista_mexico, ignore_index=True)
    
    # IMPORTANTE: Parquet requiere tipos de datos consistentes.
    # Convertimos las columnas de texto/mezcladas a string para evitar errores de esquema.
    for col in df_completo.columns:
        if df_completo[col].dtype == 'object':
            df_completo[col] = df_completo[col].astype(str)

    print(f"Guardando archivo final con {len(df_completo)} filas...")
    df_completo.to_parquet(ruta_salida, index=False, engine='pyarrow')
    
    print("-" * 30)
    print(f"¡ÉXITO! Archivo México_Completo_2020.parquet creado.")
    print(f"Total de registros apilados: {len(df_completo)}")
    print("-" * 30)
else:
    print("No se encontró información para procesar.")
