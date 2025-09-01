import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X, y, model_type='random_forest', cv=5, random_state=42):
    """
    Entrena un modelo de clasificación con validación cruzada.

    Args:
        X: Características preprocesadas.
        y: Variable objetivo.
        model_type (str): Tipo de modelo a entrenar.
        cv (int): Número de folds para validación cruzada.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        model: Modelo entrenado.
        cv_scores (dict): Puntuaciones de validación cruzada.
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Seleccionar el modelo según el tipo especificado
    if model_type == 'logistic_regression':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=random_state)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=random_state)
    else:
        raise ValueError(f"Tipo de modelo no válido: {model_type}")

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Calcular puntuaciones de validación cruzada
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calcular métricas de rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Crear informe de clasificación
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Almacenar resultados
    cv_scores = {
        'cv_accuracy': cv_accuracy.mean(),
        'cv_precision': cv_precision.mean(),
        'cv_recall': cv_recall.mean(),
        'cv_f1': cv_f1.mean(),
        'cv_roc_auc': cv_roc_auc.mean(),
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_roc_auc': roc_auc,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

    return model, cv_scores

def optimize_hyperparameters(X, y, model_type='random_forest', cv=5, random_state=42):
    """
    Optimiza los hiperparámetros del modelo mediante búsqueda en cuadrícula.

    Args:
        X: Características preprocesadas.
        y: Variable objetivo.
        model_type (str): Tipo de modelo a optimizar.
        cv (int): Número de folds para validación cruzada.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        best_model: Modelo optimizado.
        best_params (dict): Mejores hiperparámetros.
    """
    # Definir los hiperparámetros a optimizar según el tipo de modelo
    if model_type == 'logistic_regression':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=random_state)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    else:
        raise ValueError(f"Tipo de modelo no válido: {model_type}")

    # Realizar búsqueda en cuadrícula
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X, y)

    # Obtener el mejor modelo y los mejores hiperparámetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def save_model(model, path='../models/credit_model.joblib'):
    """
    Guarda el modelo entrenado en disco.

    Args:
        model: Modelo entrenado a guardar.
        path (str): Ruta donde guardar el modelo.
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    joblib.dump(model, path)
    print(f"Modelo guardado en {path}")

def load_model(path='../models/credit_model.joblib'):
    """
    Carga un modelo desde disco.

    Args:
        path (str): Ruta donde está guardado el modelo.

    Returns:
        model: Modelo cargado.
    """
    return joblib.load(path)

def plot_confusion_matrix(conf_matrix, class_names=['Rechazado', 'Aprobado']):
    """
    Visualiza la matriz de confusión.

    Args:
        conf_matrix: Matriz de confusión calculada.
        class_names (list): Nombres de las clases.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()

    return plt

def plot_feature_importance(model, feature_names):
    """
    Visualiza la importancia de las características para modelos que lo soportan.

    Args:
        model: Modelo entrenado.
        feature_names (list): Nombres de las características.

    Returns:
        plt: Objeto pyplot con la visualización.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Importancia de Características')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        return plt
    else:
        print("Este modelo no proporciona importancia de características")
