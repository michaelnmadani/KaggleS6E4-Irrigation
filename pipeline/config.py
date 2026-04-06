"""Competition configuration for Kaggle Playground Series S6E4."""

COMPETITION_SLUG = "playground-series-s6e4"
COMPETITION_NAME = "Predicting Irrigation Need"
TARGET = "Irrigation_Need"
CLASS_LABELS = ["Low", "Medium", "High"]
METRIC = "balanced_accuracy"
ID_COL = "id"

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
SUBMISSION_DIR = "outputs/submissions"

SOIL_FEATURES = [
    "Soil_Type", "Soil_pH", "Soil_Moisture", "Soil_Organic_Carbon",
    "Soil_Electrical_Conductivity"
]
WEATHER_FEATURES = [
    "Temperature", "Humidity", "Rainfall", "Sunlight_Hours", "Wind_Speed"
]
CROP_FEATURES = [
    "Crop_Type", "Growth_Stage", "Season", "Irrigation_Method", "Water_Source"
]
FIELD_FEATURES = [
    "Field_Area", "Mulching", "Prev_Irrigation_Amount", "Region"
]

ALL_FEATURES = SOIL_FEATURES + WEATHER_FEATURES + CROP_FEATURES + FIELD_FEATURES

CATEGORICAL_FEATURES = [
    "Soil_Type", "Crop_Type", "Growth_Stage", "Season",
    "Irrigation_Method", "Water_Source", "Mulching", "Region"
]
NUMERIC_FEATURES = [
    f for f in ALL_FEATURES if f not in CATEGORICAL_FEATURES
]

CV_FOLDS = 5
RANDOM_STATE = 42
