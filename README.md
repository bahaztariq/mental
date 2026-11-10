# Prédiction du risque de santé mentale chez les étudiants

Projet Data Science réalisé dans le cadre d'un brief pédagogique (YouCode — Jury Blanc). L'objectif est d'analyser un jeu de données décrivant le profil et le mode de vie d'étudiants, d'étudier les facteurs associés au bien-être psychologique, puis de construire un prototype de classification supervisée du risque de santé mentale.

> ⚠️ **Avertissement** : ce projet est strictement pédagogique. Le modèle ne doit jamais être présenté comme un outil de diagnostic, de dépistage clinique, de décision disciplinaire ou de sélection d'étudiants.

## 📋 Contexte et objectifs métier

Un établissement d'enseignement souhaite comprendre les facteurs associés aux difficultés psychologiques chez ses étudiants afin de mieux orienter ses actions de prévention et d'accompagnement :

- **Détection des risques** : identifier les étudiants présentant un risque accru de difficultés psychologiques ;
- **Prévention** : anticiper les besoins d'accompagnement ;
- **Analyse des facteurs** : comprendre les facteurs associés aux risques psychologiques ;
- **Accompagnement** : orienter les étudiants vers des ressources adaptées ;
- **Optimisation** : prioriser les actions de soutien étudiant.

## 📊 Dataset

- **Source** : [Mental Health Risk Prediction Dataset — Students (Kaggle)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
- **Fichier** : `src/mental/Data/student_mental_health.csv`
- **Taille** : 2 000 étudiants, 17 variables

| Variable | Description |
|---|---|
| `Student_ID` | Identifiant de l'étudiant |
| `Age` | Âge |
| `Gender` | Genre |
| `Year_of_Study` | Année d'étude |
| `Sleep_Hours` | Heures de sommeil par jour |
| `Study_Hours_Per_Week` | Heures d'étude par semaine |
| `GPA` | Moyenne académique |
| `Social_Media_Hours` | Heures sur les réseaux sociaux |
| `Screen_Time_Hours` | Temps d'écran |
| `Physical_Activity` | Niveau d'activité physique (Low / Moderate / High) |
| `Financial_Stress` | Stress financier (score) |
| `Family_Support` | Soutien familial (score) |
| `Attendance_Rate` | Taux d'assiduité (%) |
| `Part_Time_Job` | Emploi à temps partiel (Yes / No) |
| `Diet_Quality` | Qualité de l'alimentation (Poor / Average / Good) |
| `Previous_Mental_Health_History` | Antécédent de santé mentale (Yes / No) |
| `Mental_Health_Risk` | **Variable cible** : niveau de risque (Low / Medium / High) |

## 🧭 Démarche

Le projet couvre l'ensemble de la chaîne Data Science :

1. **Chargement et Analyse exploratoire des données (EDA)** — structure du dataset, valeurs manquantes, doublons, incohérences, outliers, statistiques descriptives, distribution de la cible, relations variables/cible, corrélations → `src/mental/1-eda.ipynb`
2. **Transformation et préparation des données** — split train/test (80/20), imputation des valeurs manquantes, traitement des outliers (IQR), encodage des variables catégorielles, gestion du déséquilibre des classes (SMOTE si nécessaire), standardisation, pipeline anti-data-leakage
3. **Entraînement et comparaison des modèles** — au moins 3 modèles de classification comparés sur les mêmes données (accuracy, précision, rappel, F1-score, matrice de confusion, ROC-AUC)
4. **Optimisation et évaluation** — hyperparameter tuning (GridSearchCV / RandomizedSearchCV), évaluation sur le jeu de test indépendant, analyse des erreurs, sauvegarde du pipeline (`.pkl` / `.joblib`)
5. **Orchestration et application** — DAG Airflow pour automatiser le pipeline, application Streamlit pour visualiser les performances et tester le modèle

## 🏗️ Structure du projet

```
mental/
├── pyproject.toml            # Configuration du projet (uv)
├── uv.lock                   # Verrou des dépendances
├── .python-version           # Version Python
├── .gitignore
└── src/
    └── mental/
        ├── BRIEF SC1.md      # Brief du projet
        ├── 1-eda.ipynb       # Notebook EDA (étape 1)
        └── Data/
            └── student_mental_health.csv
```

## 📈 Résultats

> Section à compléter après l'entraînement et l'optimisation des modèles (étapes 3 et 4) : métriques par modèle, matrice de confusion, choix final et justification.

## ⚙️ Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) comme gestionnaire de paquets :

```bash
# Cloner le dépôt
git clone <url-du-depot>
cd mental

# Installer les dépendances
uv sync
```

## ▶️ Exécution

Lancer le notebook EDA (étape 1) :

```bash
uv run jupyter lab src/mental/1-eda.ipynb
```

*(Les instructions de lancement de l'application Streamlit et du DAG Airflow seront ajoutées aux étapes 4 et 5.)*

## 📸 Captures d'écran

> Captures de l'application Streamlit à insérer après l'étape 5.

## 📄 Licence

Projet pédagogique — tous droits réservés à YouCode.
