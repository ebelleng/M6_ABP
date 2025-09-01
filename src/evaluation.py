import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import learning_curve, validation_curve

def evaluate_classification_model(model, X_test, y_test):
    """
    Evalúa un modelo de clasificación usando varias métricas.

    Args:
        model: Modelo entrenado.
        X_test: Características de prueba.
        y_test: Etiquetas reales.

    Returns:
        dict: Diccionario con las métricas de evaluación.
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # Agregar AUC-ROC si es posible
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)

    # Crear informe de clasificación
    metrics['classification_report'] = classification_report(y_test, y_pred)

    return metrics

def evaluate_regression_model(model, X_test, y_test):
    """
    Evalúa un modelo de regresión usando varias métricas.

    Args:
        model: Modelo entrenado.
        X_test: Características de prueba.
        y_test: Valores reales.

    Returns:
        dict: Diccionario con las métricas de evaluación.
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2
    }

    return metrics

def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Genera curvas de aprendizaje para evaluar el ajuste del modelo.

    Args:
        model: Modelo a evaluar.
        X: Características.
        y: Etiquetas.
        cv (int): Número de folds para validación cruzada.
        train_sizes (array): Tamaños relativos de los conjuntos de entrenamiento.

    Returns:
        plt: Objeto pyplot con la visualización.
    """
    plt.figure(figsize=(10, 6))

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Entrenamiento')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validación')

    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel('Precisión')
    plt.title('Curva de Aprendizaje')
    plt.legend(loc='best')
    plt.grid(True)

    return plt

def plot_validation_curve(model, X, y, param_name, param_range, cv=5):
    """
    Genera curvas de validación para evaluar el ajuste del modelo respecto a un hiperparámetro.

    Args:
        model: Modelo a evaluar.
        X: Características.
        y: Etiquetas.
        param_name (str): Nombre del hiperparámetro.
        param_range (array): Valores del hiperparámetro a evaluar.
        cv (int): Número de folds para validación cruzada.

    Returns:
        plt: Objeto pyplot con la visualización.
    """
    plt.figure(figsize=(10, 6))

    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='accuracy', n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Entrenamiento')
    plt.plot(param_range, test_scores_mean, 'o-', color='g', label='Validación')

    plt.xlabel(param_name)
    plt.ylabel('Precisión')
    plt.title(f'Curva de Validación - {param_name}')
    plt.legend(loc='best')
    plt.grid(True)

    return plt

def plot_roc_curve(model, X_test, y_test):
    """
    Genera una curva ROC para evaluar el rendimiento del clasificador.

    Args:
        model: Modelo entrenado.
        X_test: Características de prueba.
        y_test: Etiquetas reales.

    Returns:
        plt: Objeto pyplot con la visualización.
    """
    from sklearn.metrics import roc_curve, auc

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True)

    return plt

def plot_residuals(model, X_test, y_test):
    """
    Genera un gráfico de residuos para evaluar un modelo de regresión.

    Args:
        model: Modelo de regresión entrenado.
        X_test: Características de prueba.
        y_test: Valores reales.

    Returns:
        plt: Objeto pyplot con la visualización.
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 6))

    # Gráfico de residuos
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Valores Predichos')
    plt.ylabel('Residuos')
    plt.title('Gráfico de Residuos')
    plt.grid(True)

    # Histograma de residuos
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel('Residuos')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Residuos')
    plt.grid(True)

    plt.tight_layout()

    return plt
