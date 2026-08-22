"""
PowerGenAI - Feature Engineering Module (Phase 3 logic)
-----------------------------------------------------------
Derives the engineered features used throughout the project from the
cleaned raw columns. All division-by-zero cases are set to NaN rather
than 0 or infinity, since "undefined" is different from "zero impact".
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Available_Capacity'] = df['Monitored_Capacity'] - df['Total_Maintenance']

    df['Maintenance_Impact_Index'] = np.where(
        df['Monitored_Capacity'] > 0,
        (df['Total_Maintenance'] / df['Monitored_Capacity']) * 100,
        np.nan,
    )

    df['Programme_Achievement'] = np.where(
        df['Programme'] > 0,
        (df['Actual'] / df['Programme']) * 100,
        np.nan,
    ) if 'Actual' in df.columns else np.nan

    df['Capacity_Utilization'] = np.where(
        df['Available_Capacity'] > 0,
        (df['Actual'] / df['Available_Capacity']) * 100,
        np.nan,
    ) if 'Actual' in df.columns else np.nan

    if 'Actual' in df.columns:
        df['Generation_Difference'] = df['Actual'] - df['Programme']

    df['Forced_Maintenance_Ratio'] = np.where(
        df['Monitored_Capacity'] > 0, df['Forced_Maintenance'] / df['Monitored_Capacity'], np.nan
    )
    df['Planned_Maintenance_Ratio'] = np.where(
        df['Monitored_Capacity'] > 0, df['Planned_Maintenance'] / df['Monitored_Capacity'], np.nan
    )
    df['Other_Reason_Ratio'] = np.where(
        df['Monitored_Capacity'] > 0, df['Other_Reasons'] / df['Monitored_Capacity'], np.nan
    )

    return df


# The exact feature set (and order) the trained ML models expect as input.
MODEL_FEATURE_COLUMNS = [
    'Power_Station', 'Monitored_Capacity', 'Total_Maintenance', 'Planned_Maintenance',
    'Forced_Maintenance', 'Other_Reasons', 'Programme', 'Available_Capacity',
    'Maintenance_Impact_Index', 'Forced_Maintenance_Ratio', 'Planned_Maintenance_Ratio',
    'Other_Reason_Ratio',
]
