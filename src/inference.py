import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, RISK_ORDER, RISK_MAP

CATEGORIES = {
    "Gender": ["Female", "Male", "Non-binary"],
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
    "Part_Time_Job": ["No", "Yes"],
    "Previous_Mental_Health_History": ["No", "Yes"],
}

INV_RISK_MAP = {v: k for k, v in RISK_MAP.items()}


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Valide qu'un DataFrame contient les colonnes et valeurs requises pour la prédiction."""
    problems = []

    if df.empty:
        return ["Le fichier est vide."]

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        problems.append(f"Colonnes manquantes : {', '.join(sorted(missing))}")

    for col, allowed in CATEGORIES.items():
        if col in df.columns:
            invalid = set(df[col].dropna().unique()) - set(allowed)
            if invalid:
                problems.append(
                    f"Valeurs inattendues dans '{col}' : {', '.join(sorted(map(str, invalid)))}"
                )

    return problems


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """Effectue les prédictions et ajoute les probabilités ainsi que la classe estimée."""
    X = df[FEATURE_COLS].copy()
    preds = model.predict(X)
    probas = model.predict_proba(X)

    res = df.copy()

    # Conversion des prédictions en libellés textuels (Low, Medium, High)
    res["Risque_estime"] = [INV_RISK_MAP.get(p, p) for p in preds]

    # Ajout des probabilités par classe
    for i, cls_idx in enumerate(model.classes_):
        class_name = INV_RISK_MAP.get(cls_idx, cls_idx)
        res[f"Proba_{class_name}"] = probas[:, i]

    return res
