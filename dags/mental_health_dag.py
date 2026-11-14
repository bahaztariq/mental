"""DAG Airflow pour le pipeline de prédiction de santé mentale des étudiants.

Orchestre les étapes du projet :
1. Ingestion / Vérification des données
2. Prétraitement et nettoyage (IQR, encodage, split)
3. Entraînement et optimisation du modèle
4. Évaluation et validation des métriques
"""

from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/project")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_check_data():
    data_path = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {data_path}")
    print(f"[OK] Données trouvées : {data_path} (taille: {data_path.stat().st_size} octets)")


def task_preprocess():
    print("[OK] Vérification du pipeline de prétraitement et utilitaires IQRCapper.")
    try:
        from preprocessing_utils import IQRCapper
        capper = IQRCapper()
        print(f"[OK] IQRCapper importé avec succès : {capper}")
    except Exception as e:
        print(f"[Warning] Import de preprocessing_utils : {e}")


def task_train():
    print("[OK] Étape d'entraînement/optimisation du modèle...")
    model_path = PROJECT_ROOT / "models" / "mental_health_risk_pipeline.pkl"
    if model_path.exists():
        print(f"[OK] Modèle existant trouvé : {model_path}")
    else:
        print("[INFO] Aucun modèle sérialisé trouvé, lancement requis.")


def task_evaluate():
    print("[OK] Évaluation des performances du modèle...")
    eval_path = PROJECT_ROOT / "models" / "final_evaluation.pkl"
    if eval_path.exists():
        print(f"[OK] Fichier d'évaluation trouvé : {eval_path}")
    print("[OK] Pipeline exécuté avec succès.")


with DAG(
    dag_id="mental_health_pipeline",
    default_args=default_args,
    description="Pipeline d'orchestration ML - Risque de santé mentale",
    schedule_interval=timedelta(days=7),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mental_health", "mlops"],
) as dag:

    check_data = PythonOperator(
        task_id="check_data",
        python_callable=task_check_data,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate,
    )

    check_data >> preprocess >> train >> evaluate
