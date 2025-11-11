import os

import streamlit as st
import pandas as pd
import requests
import json

# Configuración del endpoint
AZURE_ENDPOINT = "http://14caf6ec-bc2e-4dba-aa5c-06e2c1657dc3.westus2.azurecontainer.io/score"
AZURE_KEY = os.getenv("AZURE_KEY", "")

st.set_page_config(page_title="SmartPrice", page_icon="📊", layout="wide")
st.image("./assets/sp_logo.png", width=180)
st.title("SmartPrice - Predicción de precios de casas utilizando Machine Learning 🤖🏡")

st.markdown("Subí un archivo CSV con los datos de la o las casas elegidas y conoce su precio de venta en segundos")


uploaded_file = st.file_uploader("Elegí tu CSV", type=["csv"])

if uploaded_file is not None:
    # Leer CSV
    df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(df.head())

    if st.button("Generar predicciones"):
        payload = df.to_json(orient="records")
        headers = {"Content-Type": "application/json"}
        if AZURE_KEY:
            headers["Authorization"] = f"Bearer {AZURE_KEY}"

        with st.spinner("Prediciendo 🚀"):
            response = requests.post(AZURE_ENDPOINT, data=payload, headers=headers)

        if response.status_code == 200:
            st.success("Predicciones completadas ✔")
            preds = json.loads(response.json())
            preds_df = pd.DataFrame(preds)
            cols_to_show = [
                "Id",
                "SalePrice_pred"

            ]
            # Usar .loc para evitar error si falta alguna columna
            result = preds_df[cols_to_show]
            result["SalePrice_pred"] = result["SalePrice_pred"].apply(
                lambda x: f"USD {round(x):,}"
            )

            st.write("Resultados:")
            st.dataframe(result)
        else:
            st.error(f"Error al predecir :( ❌  - Status code: {response.status_code}")
            st.text(response.text)
