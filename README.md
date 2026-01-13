# Clasificador de Nivel Socioeconómico (NSE) para México

Modelo de Machine Learning basado en ensemble que clasifica áreas geográficas según su nivel socioeconómico utilizando datos censales del INEGI (https://www.inegi.org.mx/programas/ccpv/2020/#datos_abiertos).

## 📊 Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de NSE para México usando variables demográficas y socioeconómicas del censo. El modelo predice 7 categorías socioeconómicas: **AB, C+, C, C-, D+, D y E**.

## 🎯 Rendimiento del Modelo

### Métricas Globales
- **Precisión en Test**: 71.61%
- **Precisión en Predicción Real**: 83.87%
- **Total de muestras evaluadas**: 107,887

### Matriz de Confusión
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

### Métricas por Clase

| Clase | Precisión | Recall | F1-Score | Soporte | Observaciones |
|-------|-----------|--------|----------|---------|---------------|
| AB    | 0.80      | 0.70   | 0.75     | 7,528   | Alto nivel socioeconómico |
| C     | 0.62      | 0.63   | 0.63     | 15,840  | Confusión con C+ y C- |
| C+    | 0.67      | 0.70   | 0.68     | 12,715  | Buena separación |
| C-    | 0.78      | 0.75   | 0.76     | 31,356  | Mejor desempeño |
| D     | 0.74      | 0.77   | 0.75     | 17,434  | Buen balance |
| D+    | 0.69      | 0.71   | 0.70     | 22,376  | Confusión con D y C- |
| E     | 0.69      | 0.28   | 0.40     | 638     | Clase minoritaria |

**Promedio ponderado**: Precisión 0.72, Recall 0.72, F1-Score 0.72

## 🧠 Arquitectura del Modelo

### Voting Classifier (Soft Voting)
Ensemble de 5 clasificadores con pesos optimizados:

1. **Random Forest** (peso: 3)
   - n_estimators: 200
   - max_depth: 20
   - min_samples_split: 2
   - min_samples_leaf: 2
   
2. **XGBoost** (peso: 2)
   - objective: 'multi:softmax'
   - n_estimators: 200
   - max_depth: 10
   - learning_rate: 0.1
   
3. **CatBoost** (peso: 2)
   - iterations: 300
   - learning_rate: 0.1
   - depth: 10
   
4. **AdaBoost** (peso: 1)
   - n_estimators: 200
   
5. **Extra Trees** (peso: 1)
   - n_estimators: 200
   - max_depth: 20

### Estrategia de Validación
- Split: 80% entrenamiento / 20% prueba
- Random state: 42
- Sin estandarización (mejora el rendimiento con árboles)

## 📝 Variables Utilizadas (32 features)

### Variables de Vivienda (% sobre total de viviendas)
- `vph_excsa`: Viviendas con excusado
- `vph_autom`: Viviendas con automóvil
- `vph_inter`: Viviendas con internet
- `vph_3ymasc`: Viviendas con 3 o más cuartos
- `vph_stvp`: Viviendas sin televisión de paga
- `vph_pc`: Viviendas con computadora
- `vph_cvj`: Viviendas con consola de videojuegos
- `vph_2ymasd`: Viviendas con 2 o más dormitorios
- `vph_moto`: Viviendas con motocicleta
- `vph_bici`: Viviendas con bicicleta
- `vph_lavad`: Viviendas con lavadora
- `vph_hmicro`: Viviendas con horno de microondas
- `vph_refri`: Viviendas con refrigerador
- `vph_telef`: Viviendas con teléfono
- `vph_spmvpi`: Viviendas sin ningún bien
- `vph_tv`: Viviendas con televisión
- `vph_radio`: Viviendas con radio
- `vph_1cuart`: Viviendas con 1 cuarto

### Variables de Población Económicamente Activa
- `pocupada`: Población ocupada (% sobre PEA)
- `p18ym_pb`: Población de 18 años y más con primaria básica (% sobre PEA)
- `pder_imss`: Población derechohabiente del IMSS (% sobre PEA)

### Variables de Población Total
- `pder_ss`: Población con derecho a servicios de salud (% sobre población total)
- `p15sec_co`: Población de 15 años y más con secundaria completa
- `p_60ymas`: Población de 60 años y más

### Variables Demográficas por Edad
- `p_0a2`: Población de 0 a 2 años
- `p_3a5`: Población de 3 a 5 años
- `p_6a11`: Población de 6 a 11 años
- `p_12a14`: Población de 12 a 14 años
- `p_15a17`: Población de 15 a 17 años
- `p_18a24`: Población de 18 a 24 años
- `pob15_64`: Población de 15 a 64 años

### Otras Variables
- `graproes`: Grado promedio de escolaridad

## 🚀 Uso

### 1. Entrenamiento del Modelo

```python
python modelo_entrenamiento.py
```

**Entrada requerida**:
- Carpeta con archivos CSV de entrenamiento (datos INEGI)
- Cada archivo debe incluir columna `nse` con la clasificación real
- Variables demográficas y de vivienda requeridas

**Salida**:
- `modelo_nse.pkl`: Modelo entrenado serializado
- Reporte de clasificación en consola
- Archivo Excel con predicciones

### 2. Predicción con Modelo Entrenado

```python
python prediccion_nse.py
```

**Entrada requerida**:
- `modelo_nse.pkl`: Modelo previamente entrenado
- Archivo CSV/Excel con datos a predecir

**Salida**:
- Archivo Excel con columna `nse_predicho`
- Métricas de precisión (si existe columna `nse` real)

## 📦 Dependencias

```bash
pip install pandas numpy scikit-learn catboost xgboost joblib openpyxl
```

### Versiones recomendadas
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
catboost>=1.2
xgboost>=1.7.0
joblib>=1.2.0
openpyxl>=3.0.0
```

## 📂 Estructura del Proyecto

```
proyecto_nse/
│
├── modelo_entrenamiento.py    # Script de entrenamiento
├── prediccion_nse.py           # Script de predicción
├── modelo_nse.pkl              # Modelo entrenado (generado)
├── README.md                   # Este archivo
│
├── datos/
│   ├── NSE/                    # Carpeta con CSVs de entrenamiento
│   │   ├── estado1.csv
│   │   ├── estado2.csv
│   │   └── ...
│   └── prediccion/
│       └── datos_nuevos.csv    # Datos para clasificar
│
└── resultados/
    └── predicciones.xlsx       # Salida con clasificaciones
```

## 🔧 Preprocesamiento de Datos

### Limpieza
- Eliminación de asteriscos (*) en valores numéricos
- Conversión a tipo numérico con manejo de errores
- Filtrado de valores nulos e infinitos

### Transformaciones
Todas las variables de vivienda y población se expresan como porcentajes:
- Variables de vivienda: `(valor / total_viviendas) * 100`
- Variables de PEA: `(valor / poblacion_activa) * 100`
- Variables demográficas: `(valor / poblacion_total) * 100`

### Filtrado de Categorías
Se excluyen las siguientes categorías:
- `IND`: Indeterminado
- `ND`: No disponible
- `C/S`: Con/Sin dato
- `NS`: No especificado

## 📊 Interpretación de Resultados

### Categorías NSE
- **AB**: Nivel socioeconómico alto
- **C+**: Nivel medio-alto
- **C**: Nivel medio
- **C-**: Nivel medio-bajo
- **D+**: Nivel bajo-alto
- **D**: Nivel bajo
- **E**: Nivel muy bajo

### Observaciones del Modelo
1. **Mejor desempeño**: Clases C- y AB (F1-Score > 0.75)
2. **Desafío principal**: Clase E por desbalance de datos (solo 638 muestras)
3. **Confusión común**: Entre niveles adyacentes (C, C+, C-)
4. **Fortaleza**: Alta precisión en extremos (AB y D)

## 🎓 Aplicaciones

- **Análisis de mercado**: Segmentación geográfica para estrategias comerciales
- **Políticas públicas**: Identificación de áreas prioritarias para programas sociales
- **Investigación social**: Estudios demográficos y socioeconómicos
- **Planeación urbana**: Desarrollo de infraestructura y servicios

## ⚠️ Limitaciones

1. **Desbalance de clases**: La clase E está subrepresentada
2. **Confusión entre niveles medios**: C, C+ y C- tienen características similares
3. **Dependencia de datos censales**: Requiere actualización periódica
4. **Contexto geográfico**: Entrenado con datos mexicanos (INEGI)

## 🔄 Futuras Mejoras

- [ ] Implementar técnicas de oversampling para clase E (SMOTE)
- [ ] Añadir validación cruzada estratificada
- [ ] Incluir variables geoespaciales (latitud/longitud)
- [ ] Experimentar con redes neuronales para capturar interacciones complejas
- [ ] Crear API REST para predicciones en tiempo real
- [ ] Implementar explicabilidad del modelo (SHAP values)

## 👥 Autor

**José Luis Valdez**

## 📄 Licencia

Este proyecto utiliza datos públicos del INEGI (Instituto Nacional de Estadística y Geografía de México).

## 🙏 Agradecimientos

- INEGI por proporcionar datos censales de alta calidad
- Comunidad de scikit-learn, XGBoost y CatBoost por sus excelentes herramientas

---

**Última actualización**: Enero 2026  
**Versión del modelo**: 1.0  
**Precisión en producción**: 83.87%

