from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="Modelo de Clasificación de Phishing",
    version="1.0.0"
)

# ------------------------------------------------------------
# CARGAR EL MODELO
# ------------------------------------------------------------
modelo = joblib.load("notebooks/classification_model.joblib")


@app.post("/api/v1/predecir-phishing", tags=["phishing"])
async def predecir(
    num_words: float,
    num_unique_words: float,
    num_stopwords: float,
    num_links: float,
    num_unique_domains: float,
    num_email_addresses: float,
    num_spelling_errors: float,
    num_urgent_keywords: float
):
    # Construir el diccionario con los valores recibidos
    datos_dict = {
        "num_words": num_words,
        "num_unique_words": num_unique_words,
        "num_stopwords": num_stopwords,
        "num_links": num_links,
        "num_unique_domains": num_unique_domains,
        "num_email_addresses": num_email_addresses,
        "num_spelling_errors": num_spelling_errors,
        "num_urgent_keywords": num_urgent_keywords
    }

    try:
        # Convertir a DataFrame
        df = pd.DataFrame([datos_dict])

        # Predecir con el modelo
        prediccion = modelo.predict(df)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediccion": int(prediccion[0])}
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
