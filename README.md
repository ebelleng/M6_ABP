# Proyecto de Evaluación Crediticia - Alke Wallet

## Descripción del Proyecto

Este proyecto implementa un sistema de evaluación crediticia automatizada para Alke Wallet, una fintech emergente. El sistema utiliza técnicas de machine learning para predecir si un cliente es apto para recibir servicios financieros, basándose en su historial crediticio y características financieras.

## Estructura del Proyecto

```
M6_ABP/
├── api/                  # Implementación de la API para el despliegue del modelo
│   ├── app.py           # Aplicación FastAPI
│   └── requirements.txt  # Dependencias para la API
├── data/                 # Datos utilizados en el proyecto
├── docs/                 # Documentación del proyecto
│   ├── api_documentation.md  # Documentación de la API
│   └── api_documentation.pdf # Versión PDF de la documentación
├── models/               # Modelos entrenados y preprocesadores
├── notebooks/            # Jupyter notebooks para análisis y desarrollo
│   ├── 01_exploratory_analysis.ipynb  # Análisis exploratorio de datos
│   ├── 02_preprocessing.ipynb         # Preprocesamiento de datos
│   ├── 03_model_training.ipynb        # Entrenamiento de modelos
│   └── 04_model_evaluation.ipynb      # Evaluación del modelo
└── src/                  # Código fuente modular
    ├── evaluation.py     # Funciones para evaluar modelos
    ├── models.py         # Funciones para entrenar modelos
    └── preprocessing.py  # Funciones para preprocesar datos
```

## Flujo de Trabajo del Proyecto

1. **Análisis Exploratorio de Datos**:
   - Análisis de distribuciones y relaciones entre variables
   - Identificación de correlaciones con la variable objetivo
   - Detección de valores atípicos y patrones importantes

2. **Preprocesamiento de Datos**:
   - Tratamiento de valores atípicos mediante recorte (capping)
   - Estandarización de variables numéricas
   - Codificación one-hot para variables categóricas
   - División en conjuntos de entrenamiento y prueba

3. **Entrenamiento de Modelos**:
   - Evaluación de múltiples algoritmos (Regresión Logística, Random Forest, Gradient Boosting, SVM)
   - Validación cruzada para medir rendimiento
   - Optimización de hiperparámetros mediante búsqueda en cuadrícula
   - Selección del mejor modelo basado en métricas clave

4. **Evaluación del Modelo**:
   - Análisis detallado del rendimiento con múltiples métricas
   - Análisis de la matriz de confusión y casos específicos
   - Evaluación de curvas ROC y precisión-recall
   - Análisis de calibración y ajuste del umbral de decisión

5. **Despliegue del Modelo**:
   - Implementación de una API REST con FastAPI
   - Endpoints para realizar predicciones en tiempo real
   - Documentación completa de la API

## Principales Hallazgos

- El modelo Random Forest mostró el mejor rendimiento general con un F1-Score de 0.8768 y ROC AUC de 0.9324.
- Las características más importantes para la predicción son:
  - Ratio deuda-ingreso
  - Duración del historial crediticio
  - Número de pagos tardíos
  - Ingresos mensuales
- El umbral óptimo de decisión se estableció en 0.475, ligeramente inferior al estándar de 0.5, para maximizar el F1-Score.
- El modelo muestra buena calibración, lo que permite interpretar las probabilidades de manera confiable.

## Tecnologías Utilizadas

- **Lenguaje**: Python 3.9+
- **Análisis de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **API**: FastAPI
- **Servidor**: Uvicorn
- **Documentación**: Markdown, PDF

## Cómo Ejecutar el Proyecto

### Requisitos Previos
- Python 3.9 o superior
- pip (gestor de paquetes de Python)

### Configuración del Entorno

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/ebelleng/M6_ABP.git
   cd M6_ABP
   ```

2. Instalar dependencias:
   ```bash
   pip install -r api/requirements.txt
   ```

### Ejecución de Notebooks

Los notebooks se pueden ejecutar en orden secuencial:

1. Análisis Exploratorio de Datos
2. Preprocesamiento de Datos
3. Entrenamiento de Modelos
4. Evaluación del Modelo

```bash
jupyter notebook notebooks/
```

### Iniciar la API

```bash
cd api
uvicorn app:app --reload
```

La API estará disponible en: http://localhost:8000

Documentación interactiva: http://localhost:8000/docs

## Autor

- Etienne Bellenger - [ebelleng](https://github.com/ebelleng)
