import json
import os
import pandas as pd, featuretools as ft, numpy as np
import traceback
from azureml.core.model import Model
import joblib

model = None
feature_defs = None


def init():
    global model, feature_defs
    try:
        artifacts_dir = Model.get_model_path("houses-price-predictor-model")
        # defs_path = Model.get_model_path("feature_defs.pkl")
        # feature_defs = joblib.load(defs_path)
        model_path = os.path.join(artifacts_dir, "houseprice_final_model.pkl")
        if model_path.endswith(".pkl"):
            model_path = model_path[:-4]


        feature_defs = joblib.load(os.path.join(artifacts_dir, "feature_defs.pkl"))
        try:
            from pycaret.regression import load_model as py_load_model
            model = py_load_model(model_path)
            return
        except Exception:
            # fallback to joblib/pickle for plain pickles
            pass
        model = joblib.load(model_path)
    except Exception:
        print("Model load failed:")
        traceback.print_exc()
        raise


def run(raw_data):
    try:
        base_data = pd.DataFrame(json.loads(raw_data))

        if "Id" not in base_data.columns:
            base_data = base_data.reset_index(
                drop=False).rename(columns={"index": "row_id"})
            index_col = "row_id"
        else:
            index_col = "Id"
        # Crear EntitySet
        es = ft.EntitySet(id="houses")
        es = es.add_dataframe(
            dataframe_name="properties",
            dataframe=base_data,
            index=index_col,
            make_index=False
        )

        # Generar features con las definiciones guardadas
        feature_matrix = ft.calculate_feature_matrix(
            features=feature_defs,
            entityset=es
        ).reset_index(drop=False)

        # Limpieza rápida
        num_cols = feature_matrix.select_dtypes(include=[float, int]).columns
        feature_matrix[num_cols] = feature_matrix[num_cols].replace(
            [np.inf, -np.inf], np.nan)
        for c in num_cols:
            if feature_matrix[c].isna().any():
                feature_matrix[c] = feature_matrix[c].fillna(
                    feature_matrix[c].median())
                
        # If model is a PyCaret pipeline/estimator, use predict_model
        try:
            from pycaret.regression import predict_model as py_predict
            preds_df = py_predict(model, data=feature_matrix)
            preds_df["SalePrice_pred"] = np.expm1(preds_df["prediction_label"])
            # convert prediction column(s) to JSON
            return preds_df.to_json(orient="records")
        except Exception:
            # fallback to scikit-learn style predict
            preds = model.predict(feature_matrix)
            return json.dumps({"predictions": preds.tolist()})
    except Exception:
        print("Scoring failed:")
        traceback.print_exc()
        return json.dumps({"error": "scoring failed, check logs"})
