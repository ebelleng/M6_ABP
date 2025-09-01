# Documentación de la API de Alke Wallet

## Descripción General

Esta API proporciona un servicio de evaluación crediticia automatizada para nuevos usuarios de Alke Wallet. Utiliza un modelo de machine learning entrenado con datos históricos para predecir si un usuario será considerado apto para acceder a servicios financieros.

## Información Técnica

- **Framework**: FastAPI
- **Versión**: 1.0.0
- **Base URL**: `http://localhost:8000` (desarrollo local)

## Endpoints

### Health Check

Verifica que la API esté funcionando correctamente.

- **URL**: `/health`
- **Método**: GET
- **Respuesta Exitosa**:
  - **Código**: 200 OK
  - **Contenido**:
    ```json
    {
      "status": "OK",
      "message": "API funcionando correctamente"
    }
    ```

### Predicción de Aptitud Crediticia

Evalúa a un usuario basado en sus características financieras y determina si es apto para recibir servicios financieros.

- **URL**: `/predict`
- **Método**: POST
- **Cuerpo de la Solicitud**:
  ```json
  {
    "age": 35,
    "income": 65000.0,
    "employment_years": 5.5,
    "debt_to_income_ratio": 0.3,
    "credit_history_length": 7,
    "num_credit_accounts": 3,
    "num_late_payments": 1,
    "has_mortgage": true,
    "has_auto_loan": false,
    "has_credit_card": true
  }
  ```
- **Respuesta Exitosa**:
  - **Código**: 200 OK
  - **Contenido**:
    ```json
    {
      "prediction": 1,
      "probability": 0.85,
      "credit_status": "Aprobado"
    }
    ```
- **Respuesta de Error**:
  - **Código**: 422 Unprocessable Entity
    - Ocurre cuando los datos de entrada no cumplen con el esquema esperado.
  - **Código**: 500 Internal Server Error
    - Ocurre cuando hay un error en el proceso de predicción.

## Esquema de Datos

### Entrada (UserInput)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| age | integer | Edad del usuario en años |
| income | number | Ingreso anual en pesos |
| employment_years | number | Años de empleo en el trabajo actual |
| debt_to_income_ratio | number | Relación deuda/ingreso (0-1) |
| credit_history_length | integer | Años de historial crediticio |
| num_credit_accounts | integer | Número de cuentas de crédito activas |
| num_late_payments | integer | Número de pagos tardíos en los últimos 2 años |
| has_mortgage | boolean | Si el usuario tiene una hipoteca |
| has_auto_loan | boolean | Si el usuario tiene un préstamo de auto |
| has_credit_card | boolean | Si el usuario tiene tarjeta de crédito |

### Salida (PredictionOutput)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| prediction | integer | Predicción (0: Rechazado, 1: Aprobado) |
| probability | number | Probabilidad de aprobación (0-1) |
| credit_status | string | Estado crediticio ("Aprobado" o "Rechazado") |

## Cómo Utilizar la API

### Ejemplo con cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 65000.0,
    "employment_years": 5.5,
    "debt_to_income_ratio": 0.3,
    "credit_history_length": 7,
    "num_credit_accounts": 3,
    "num_late_payments": 1,
    "has_mortgage": true,
    "has_auto_loan": false,
    "has_credit_card": true
  }'
```

### Ejemplo con Python (Requests)

```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
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

response = requests.post(url, json=data)
result = response.json()
print(result)
```

## Ejecución Local

Para ejecutar la API localmente:

1. Asegúrate de tener instaladas todas las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Inicia el servidor:
   ```bash
   cd api
   uvicorn app:app --reload
   ```

3. Accede a la documentación interactiva:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
