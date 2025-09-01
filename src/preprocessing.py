import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def create_preprocessor(categorical_features, numerical_features):
    """
    Crea un transformador para preprocesar datos categóricos y numéricos.

    Args:
        categorical_features (list): Lista de columnas categóricas.
        numerical_features (list): Lista de columnas numéricas.

    Returns:
        ColumnTransformer: Preprocesador configurado.
    """
    # Pipeline para características numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinación de transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def preprocess_data(df, categorical_features, numerical_features, target=None, preprocessor=None, train=False):
    """
    Preprocesa los datos según las características especificadas.

    Args:
        df (DataFrame): DataFrame a preprocesar.
        categorical_features (list): Lista de columnas categóricas.
        numerical_features (list): Lista de columnas numéricas.
        target (str, optional): Nombre de la columna objetivo.
        preprocessor (ColumnTransformer, optional): Preprocesador preentrenado.
        train (bool): Indica si se está preprocesando datos de entrenamiento.

    Returns:
        tuple: (X, y) si hay target, o (X, None) si no hay target, y el preprocesador.
    """
    # Seleccionar características y objetivo
    X = df.copy()
    y = None

    if target and target in X.columns:
        y = X[target].copy()
        X = X.drop(target, axis=1)

    # Si estamos en modo entrenamiento o no se proporciona un preprocesador, crear uno nuevo
    if train or preprocessor is None:
        preprocessor = create_preprocessor(categorical_features, numerical_features)
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    return X_transformed, y, preprocessor

def save_preprocessor(preprocessor, path='../models/preprocessor.joblib'):
    """
    Guarda el preprocesador en disco.

    Args:
        preprocessor (ColumnTransformer): Preprocesador a guardar.
        path (str): Ruta donde guardar el preprocesador.
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    joblib.dump(preprocessor, path)
    print(f"Preprocesador guardado en {path}")

def load_preprocessor(path='../models/preprocessor.joblib'):
    """
    Carga un preprocesador desde disco.

    Args:
        path (str): Ruta donde está guardado el preprocesador.

    Returns:
        ColumnTransformer: Preprocesador cargado.
    """
    return joblib.load(path)
