from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn
import os
from typing import List, Dict, Any, Optional

# Crear la aplicación FastAPI
app = FastAPI(
    title="Alke Wallet - API de Evaluación Crediticia",
    description="API para predecir la aptitud crediticia de usuarios",
    version="1.0.0"
)

# Definir el directorio base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_model.joblib")

# Cargar el modelo y el preprocesador
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.joblib"))
    print("Modelo y preprocesador cargados correctamente")
except Exception as e:
    print(f"Error al cargar el modelo o preprocesador: {str(e)}")
    # Crear valores por defecto para desarrollo
    model = None
    preprocessor = None

# Definir el modelo de entrada (ajustar según las características requeridas)
class UserInput(BaseModel):
    # Ejemplo de campos (ajustar según el modelo entrenado)
    age: int
    income: float
    employment_years: float
    debt_to_income_ratio: float
    credit_history_length: int
    num_credit_accounts: int
    num_late_payments: int
    has_mortgage: bool
    has_auto_loan: bool
    has_credit_card: bool

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 65000.0,
                "employment_years": 5.5,
                "debt_to_income_ratio": 0.3,
                "credit_history_length": 7,
                "num_credit_accounts": 3,
                "num_late_payments": 1,
                "has_mortgage": True,
                "has_auto_loan": False,
                "has_credit_card": True
            }
        }

# Definir el modelo de salida
class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    credit_status: str

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "credit_status": "Aprobado"
            }
        }

# Definir el endpoint de salud
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "OK", "message": "API funcionando correctamente"}

# Definir el endpoint de predicción
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(user_data: UserInput):
    try:
        # Convertir la entrada a dataframe
        input_df = pd.DataFrame([user_data.dict()])

        # Preprocesar y predecir con el modelo
        if model is not None and preprocessor is not None:
            # Preprocesar los datos
            X = preprocessor.transform(input_df)

            # Realizar la predicción
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
        else:
            # Simulación de predicción para desarrollo
            prediction = 1  # 0: Rechazado, 1: Aprobado
            probability = 0.85

        # Determinar el estado crediticio
        credit_status = "Aprobado" if prediction == 1 else "Rechazado"

        return PredictionOutput(
            prediction=prediction,
            probability=float(probability),
            credit_status=credit_status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

# Endpoint para documentación de la API
@app.get("/", tags=["Documentation"])
def read_root():
    return {
        "message": "Bienvenido a la API de Evaluación Crediticia de Alke Wallet",
        "documentation": "/docs",
        "redoc": "/redoc"
    }

# Si se ejecuta directamente este script
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
