import joblib
import pandas as pd
from .config import MODEL_PATH, FEATURES_PATH

def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        features = [line.strip() for line in f]
    return model, features

def predict(model, df_features):
    preds_proba = model.predict_proba(df_features)[:, 1]
    preds_class = model.predict(df_features)
    return preds_proba, preds_class
