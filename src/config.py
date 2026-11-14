from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "student_mental_health.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Modèle sérialisé : vérification du chemin présent
if (ROOT_DIR / "models" / "mental_health_risk_pipeline.pkl").exists():
    MODEL_PATH = ROOT_DIR / "models" / "mental_health_risk_pipeline.pkl"
else:
    MODEL_PATH = ROOT_DIR / "models" / "pipeline.joblib"

METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
TRAIN_INFO_PATH = ROOT_DIR / "models" / "train_info.json"
FIG_DIR = ROOT_DIR / "reports" / "figures"

TARGET = "Mental_Health_Risk"
ID_COL = "Student_ID"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_F1_MACRO = 0.60

RISK_ORDER = ["Low", "Medium", "High"]
RISK_MAP = {"Low": 0, "Medium": 1, "High": 2}

CONT_COLS = [
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Attendance_Rate",
]
OTHER_NUM_COLS = ["Age", "Year_of_Study", "Financial_Stress", "Family_Support"]

ORDINAL_COLS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
}
BINARY_COLS = ["Part_Time_Job", "Previous_Mental_Health_History"]
NOMINAL_COLS = ["Gender"]

FEATURE_COLS = CONT_COLS + OTHER_NUM_COLS + list(ORDINAL_COLS) + BINARY_COLS + NOMINAL_COLS
