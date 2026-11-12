"""Utilitaires de prétraitement pour le projet de prédiction du risque de santé mentale.

Ce module contient le transformateur custom `IQRCapper`, défini dans un fichier
(et non dans le notebook) pour que la sérialisation joblib/pickle fonctionne
lors du rechargement du pipeline (app Streamlit, DAG Airflow, etc.).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorisation (clipping) des valeurs aberrantes selon la règle de l'IQR.

    À l'`fit` (sur le train uniquement), on calcule pour chaque colonne les
    bornes [Q1 - 1.5*IQR ; Q3 + 1.5*IQR]. À la `transform`, les valeurs en
    dehors de ces bornes sont ramenées à la borne la plus proche.

    Justification du choix : les outliers détectés à l'EDA sont des valeurs
    extrêmes mais plausibles (aucune erreur de saisie), on ne supprime donc
    pas les lignes — on limite simplement leur influence. Les bornes sont
    calculées sur le train seul pour éviter tout data leakage.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "lower_bounds_"):
            raise RuntimeError("IQRCapper doit être fit avant transform.")
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)
