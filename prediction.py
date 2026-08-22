"""
PowerGenAI - Prediction Module
------------------------------------
Loads the saved best model (Random Forest, chosen in Phase 8) and provides
a single-record prediction function used by the Streamlit "Generation
Prediction" page. Also includes the tree-path contribution decomposition
used as the SHAP substitute for individual prediction explanations.
"""

import os
import numpy as np
import pandas as pd
import joblib

from .features import MODEL_FEATURE_COLUMNS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Median fallback values for ratio/index features when a station's
# Monitored_Capacity is zero (matches the Phase 7 training-time imputation).
DEFAULT_FILL_VALUES = {
    'Maintenance_Impact_Index': 15.0,
    'Forced_Maintenance_Ratio': 0.05,
    'Planned_Maintenance_Ratio': 0.03,
    'Other_Reason_Ratio': 0.01,
}


def load_model(name: str = 'Random_Forest'):
    path = os.path.join(MODELS_DIR, f'{name}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Run Phase 7 training first.")
    return joblib.load(path)


def build_feature_row(power_station: str, monitored_capacity: float, programme: float,
                       planned_maintenance: float, forced_maintenance: float,
                       other_reasons: float) -> pd.DataFrame:
    """Build a single-row dataframe of derived features from raw user inputs,
    matching the Phase 3 feature engineering logic exactly."""

    total_maintenance = planned_maintenance + forced_maintenance + other_reasons
    available_capacity = monitored_capacity - total_maintenance

    if monitored_capacity > 0:
        mii = (total_maintenance / monitored_capacity) * 100
        forced_ratio = forced_maintenance / monitored_capacity
        planned_ratio = planned_maintenance / monitored_capacity
        other_ratio = other_reasons / monitored_capacity
    else:
        mii = DEFAULT_FILL_VALUES['Maintenance_Impact_Index']
        forced_ratio = DEFAULT_FILL_VALUES['Forced_Maintenance_Ratio']
        planned_ratio = DEFAULT_FILL_VALUES['Planned_Maintenance_Ratio']
        other_ratio = DEFAULT_FILL_VALUES['Other_Reason_Ratio']

    row = pd.DataFrame([{
        'Power_Station': power_station,
        'Monitored_Capacity': monitored_capacity,
        'Total_Maintenance': total_maintenance,
        'Planned_Maintenance': planned_maintenance,
        'Forced_Maintenance': forced_maintenance,
        'Other_Reasons': other_reasons,
        'Programme': programme,
        'Available_Capacity': available_capacity,
        'Maintenance_Impact_Index': mii,
        'Forced_Maintenance_Ratio': forced_ratio,
        'Planned_Maintenance_Ratio': planned_ratio,
        'Other_Reason_Ratio': other_ratio,
    }])
    return row[MODEL_FEATURE_COLUMNS]


def predict(model, feature_row: pd.DataFrame) -> float:
    return float(model.predict(feature_row)[0])


def risk_level(predicted_actual: float, programme: float) -> str:
    """Simple, transparent risk classification based on predicted
    programme achievement. Software-defined thresholds, not a regulatory
    standard."""
    if programme <= 0:
        return "UNKNOWN (no programme value to compare against)"
    achievement = (predicted_actual / programme) * 100
    if achievement >= 95:
        return "LOW"
    elif achievement >= 80:
        return "MEDIUM"
    elif achievement >= 60:
        return "HIGH"
    else:
        return "CRITICAL"


# ---- Tree-path contribution decomposition (SHAP substitute, Phase 10) ----

def _tree_path_contributions(tree, x_row, n_features):
    tree_ = tree.tree_
    node = 0
    contrib = np.zeros(n_features)
    bias = tree_.value[0][0][0]
    current_value = bias
    while tree_.children_left[node] != tree_.children_right[node]:
        feat = tree_.feature[node]
        thresh = tree_.threshold[node]
        next_node = tree_.children_left[node] if x_row[feat] <= thresh else tree_.children_right[node]
        next_value = tree_.value[next_node][0][0]
        contrib[feat] += (next_value - current_value)
        current_value = next_value
        node = next_node
    return bias, contrib


def explain_prediction(pipeline, feature_row: pd.DataFrame):
    """Returns (bias, contributions_dict, prediction) for a Random-Forest
    pipeline, decomposing the prediction into base value + per-feature
    contributions that sum exactly to the model's output."""
    prep = pipeline.named_steps['prep']
    rf = pipeline.named_steps['model']

    x_encoded = np.asarray(prep.transform(feature_row))[0]
    n_features = len(x_encoded)

    total_bias = 0.0
    total_contrib = np.zeros(n_features)
    for est in rf.estimators_:
        b, c = _tree_path_contributions(est, x_encoded, n_features)
        total_bias += b
        total_contrib += c
    n = len(rf.estimators_)
    bias = total_bias / n
    contrib = total_contrib / n

    # ColumnTransformer puts the encoded categorical column(s) first, then
    # passthrough numeric columns in their original order — which is exactly
    # MODEL_FEATURE_COLUMNS since Power_Station is listed first there too.
    contrib_dict = dict(zip(MODEL_FEATURE_COLUMNS, contrib))
    prediction = bias + contrib.sum()
    return bias, contrib_dict, prediction
