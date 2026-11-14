import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import MODEL_PATH, METRICS_PATH, RISK_ORDER, FEATURE_COLS
from src.inference import CATEGORIES, predict_dataframe, validate_dataframe

MAX_ROWS = 10_000
FR = {
    "Low": "Faible",
    "Medium": "Moyen",
    "High": "Élevé",
    "Moderate": "Modérée",
    "Poor": "Mauvaise",
    "Average": "Moyenne",
    "Good": "Bonne",
    "No": "Non",
    "Yes": "Oui",
    "Female": "Femme",
    "Male": "Homme",
    "Non-binary": "Non-binaire",
}
fr = lambda v: FR.get(v, v)

st.set_page_config(page_title="Risque étudiant - prototype", page_icon="🎓", layout="wide")


@st.cache_resource
def _load_model(mtime: float):
    return joblib.load(MODEL_PATH)


@st.cache_data
def _load_metrics(mtime: float):
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_model():
    if not MODEL_PATH.exists():
        return None, (
            "Modèle introuvable. Lance le notebook 04 ou déclenche le DAG Airflow "
            "`mental_health_risk_pipeline_dag` pour le créer."
        )
    try:
        return _load_model(MODEL_PATH.stat().st_mtime), None
    except Exception as exc:
        return None, (
            f"Impossible de charger le modèle ({type(exc).__name__}). Cause fréquente : "
            "version de scikit-learn différente de celle de l'entraînement. "
            "Relance l'entraînement dans cet environnement."
        )


def get_metrics():
    if not METRICS_PATH.exists():
        return None, "Métriques introuvables. Lance le notebook 04 ou le DAG Airflow."
    try:
        return _load_metrics(METRICS_PATH.stat().st_mtime), None
    except (OSError, json.JSONDecodeError):
        return None, "Le fichier de métriques est illisible. Relance l'évaluation."


# --------------------------------------------------------------------------- en-tête
st.title("🎓 Risque de santé mentale étudiant : prototype pédagogique")
st.warning(
    "**Usage strictement pédagogique.** Ce modèle, entraîné sur des données **simulées**, n'est ni un "
    "outil de diagnostic, ni de dépistage clinique, ni de décision disciplinaire, ni de sélection "
    "d'étudiants. Il ne remplace pas l'avis d'un professionnel."
)
tab_perf, tab_test, tab_batch, tab_about = st.tabs(
    ["📊 Performances", "🧪 Tester un profil", "📁 Fichier CSV", "ℹ️ Limites"]
)

model, model_error = get_model()

# --------------------------------------------------------------------------- onglet 1
with tab_perf:
    metrics, err = get_metrics()
    if err:
        st.error(err)
    else:
        st.caption(
            f"Modèle : {metrics.get('model', '?')} · évalué sur {metrics.get('n_test', '?')} "
            "étudiants du jeu de test (jamais vus à l'entraînement)"
        )
        cols = st.columns(5)
        cols[0].metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
        cols[1].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[2].metric("Précision (macro)", f"{metrics['precision_macro']:.3f}")
        cols[3].metric("Rappel (macro)", f"{metrics['recall_macro']:.3f}")
        cols[4].metric("ROC-AUC (ovr)", f"{metrics['roc_auc_ovr']:.3f}")
        if "cv_f1_macro" in metrics:
            st.caption(f"F1 en validation croisée sur le train : {metrics['cv_f1_macro']:.3f}")

        st.subheader("Matrice de confusion")
        cm = np.array(metrics["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        labels = [fr(c) for c in metrics.get("classes", RISK_ORDER)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[0],
            xticklabels=labels,
            yticklabels=labels,
        )
        axes[0].set_title("Effectifs")
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=axes[1],
            xticklabels=labels,
            yticklabels=labels,
        )
        axes[1].set_title("Normalisée par ligne (rappel)")
        for ax in axes:
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Résultats par classe")
        per_class = (
            pd.DataFrame(metrics["per_class"])
            .T.rename(
                columns={
                    "precision": "Précision",
                    "recall": "Rappel",
                    "f1": "F1",
                    "support": "Effectif",
                }
            )
        )
        per_class.index = [fr(i) for i in per_class.index]
        st.dataframe(per_class.round(3))

# --------------------------------------------------------------------------- onglet 2
with tab_test:
    st.subheader("Estimer le niveau de risque d'un profil fictif")
    if model is None:
        st.error(model_error)
    else:
        with st.form("single_profile"):
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Âge", 16, 40, 21)
            year = c1.selectbox("Année d'étude", [1, 2, 3, 4, 5], index=1)
            gender = c1.selectbox("Genre", CATEGORIES["Gender"], format_func=fr)
            sleep = c1.slider("Sommeil (h par nuit)", 3.0, 10.5, 6.8, 0.1)
            study = c2.slider("Étude (h par semaine)", 0.0, 60.0, 20.0, 0.5)
            gpa = c2.slider("GPA (sur 4)", 1.0, 4.0, 2.6, 0.01)
            social = c2.slider("Réseaux sociaux (h par jour)", 0.0, 12.0, 6.0, 0.1)
            screen = c2.slider("Temps d'écran total (h par jour)", 0.0, 16.0, 8.5, 0.1)
            finance = c3.slider("Stress financier (1 à 10)", 1.0, 10.0, 6.0, 0.1)
            family = c3.slider("Soutien familial (1 à 10)", 1.0, 10.0, 6.5, 0.1)
            attendance = c3.slider("Assiduité (%)", 30.0, 100.0, 85.0, 1.0)
            activity = c3.selectbox(
                "Activité physique", CATEGORIES["Physical_Activity"], index=1, format_func=fr
            )
            diet = c3.selectbox(
                "Qualité de l'alimentation", CATEGORIES["Diet_Quality"], index=1, format_func=fr
            )
            job = c3.selectbox("Emploi à temps partiel", CATEGORIES["Part_Time_Job"], format_func=fr)
            history = c3.selectbox(
                "Antécédents de santé mentale",
                CATEGORIES["Previous_Mental_Health_History"],
                format_func=fr,
            )
            submitted = st.form_submit_button("Estimer")

        if submitted:
            if screen < social:
                st.warning(
                    "Le temps d'écran total est inférieur au temps sur les réseaux sociaux, ce qui "
                    "est impossible. Le modèle traitera le temps d'écran comme inconnu."
                )
            row = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Gender": gender,
                        "Year_of_Study": year,
                        "Sleep_Hours": sleep,
                        "Study_Hours_Per_Week": study,
                        "GPA": gpa,
                        "Social_Media_Hours": social,
                        "Screen_Time_Hours": screen,
                        "Physical_Activity": activity,
                        "Financial_Stress": finance,
                        "Family_Support": family,
                        "Attendance_Rate": attendance,
                        "Part_Time_Job": job,
                        "Diet_Quality": diet,
                        "Previous_Mental_Health_History": history,
                    }
                ]
            )
            try:
                result = predict_dataframe(model, row)
            except Exception as exc:
                st.error(f"Prédiction impossible : {exc}")
            else:
                label = result.loc[0, "Risque_estime"]
                st.info(f"Niveau estimé par le prototype : **{fr(label)}**")
                probs = pd.DataFrame(
                    {"Probabilité": [result.loc[0, f"Proba_{c}"] for c in RISK_ORDER]},
                    index=[fr(c) for c in RISK_ORDER],
                )
                st.bar_chart(probs)
                if label == "High":
                    st.caption(
                        "Ceci n'est pas un diagnostic. En cas de difficultés, un service de soutien "
                        "étudiant ou un professionnel de santé peut aider."
                    )

# --------------------------------------------------------------------------- onglet 3
SAMPLE = pd.DataFrame(
    [
        [21, "Female", 2, 6.8, 20.0, 2.6, 6.0, 8.5, "Moderate", 6.0, 6.5, 85.0, "No", "Average", "No"],
        [19, "Male", 1, 5.2, 12.0, 2.1, 7.5, 10.0, "Low", 8.0, 4.0, 70.0, "Yes", "Poor", "Yes"],
    ],
    columns=FEATURE_COLS,
)

with tab_batch:
    st.subheader("Prédictions sur un fichier CSV")
    st.caption(
        "Colonnes attendues : "
        + ", ".join(FEATURE_COLS)
        + ". Les colonnes en plus (Student_ID, cible) sont ignorées. "
        "Les valeurs manquantes sont acceptées (le modèle les impute)."
    )
    st.download_button(
        "Télécharger un exemple de CSV",
        SAMPLE.to_csv(index=False),
        "exemple_etudiants.csv",
        "text/csv",
    )
    if model is None:
        st.error(model_error)
    else:
        uploaded = st.file_uploader("Choisir un fichier CSV", type="csv")
        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
            except Exception:
                data = None
                st.error("Fichier illisible : vérifie qu'il s'agit bien d'un CSV valide.")
            if data is not None:
                problems = (
                    [f"Le fichier dépasse {MAX_ROWS} lignes."]
                    if len(data) > MAX_ROWS
                    else validate_dataframe(data)
                )
                if problems:
                    st.error("Fichier invalide :")
                    for p in problems:
                        st.write(f"- {p}")
                else:
                    result = predict_dataframe(model, data)
                    st.success(f"{len(result)} profils traités.")
                    counts = result["Risque_estime"].value_counts().reindex(RISK_ORDER).fillna(0)
                    counts.index = [fr(c) for c in counts.index]
                    st.bar_chart(counts)
                    st.dataframe(result)
                    st.download_button(
                        "Télécharger les résultats",
                        result.to_csv(index=False),
                        "predictions.csv",
                        "text/csv",
                    )

# --------------------------------------------------------------------------- onglet 4
with tab_about:
    st.markdown(
        """
### Ce que ce prototype est
Un exercice pédagogique de classification supervisée (régression logistique ou SVM) qui estime un niveau de
risque (Faible, Moyen, Élevé) à partir de variables de mode de vie : sommeil, études, stress, soutien…

### Limites à connaître
1. **Données simulées** : les performances ne se transposent pas à la réalité.
2. Le modèle reproduit des **étiquettes simulées** ; il ne détecte **pas** une détresse psychologique réelle.
3. Données **auto-déclarées** : biais de réponse possibles.
4. Jeu de test de **400 étudiants** : chaque métrique a une incertitude de quelques points.
5. La classe intermédiaire (**Moyen**) est la plus difficile à distinguer.
6. Le groupe **Non-binaire** est très petit : ses résultats sont peu fiables statistiquement.
7. **Association n'est pas causalité** : une variable liée au risque n'en est pas forcément la cause.

### Ce que ce prototype ne doit jamais servir à faire
Diagnostiquer, dépister, sanctionner, orienter une décision disciplinaire ou sélectionner des étudiants.
"""
    )
