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
    "Soil_Type", "Soil_pH", "Soil_Moisture", "Organic_Carbon",
    "Electrical_Conductivity"
]
WEATHER_FEATURES = [
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh"
]
CROP_FEATURES = [
    "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type", "Water_Source"
]
FIELD_FEATURES = [
    "Field_Area_hectare", "Mulching_Used", "Previous_Irrigation_mm", "Region"
]

ALL_FEATURES = SOIL_FEATURES + WEATHER_FEATURES + CROP_FEATURES + FIELD_FEATURES

CATEGORICAL_FEATURES = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"
]
NUMERIC_FEATURES = [
    f for f in ALL_FEATURES if f not in CATEGORICAL_FEATURES
]

CV_FOLDS = 5
CV_FOLDS_V11 = 10  # V11+: 10-fold for more stable OOF estimates
RANDOM_STATE = 42
