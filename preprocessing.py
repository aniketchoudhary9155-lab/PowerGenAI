"""
PowerGenAI - Preprocessing Module (Phase 2 logic)
----------------------------------------------------
Loads the raw CEA-style power generation Excel export and applies the
cleaning steps validated in Phase 1/2 of the project:
  - Fixes source column-name typos
  - Removes the unidentified "0" placeholder Power Station rows
  - Removes state/UT aggregate rows (different granularity than individual stations)
  - Removes exact duplicate rows
"""

import pandas as pd

RAW_COLUMN_MAP = {
    'Monitored Cap.(MW)': 'Monitored_Capacity',
    'Total Cap. Under Maintenace (MW)': 'Total_Maintenance',
    'Planned Maintanence (MW)': 'Planned_Maintenance',
    'Forced Maintanence(MW)': 'Forced_Maintenance',
    'Other Reasons (MW)': 'Other_Reasons',
    'Programme (MW)': 'Programme',
    'Actual (MW)': 'Actual',
    'Excess(+) / Shortfall (-) (MW)': 'Excess_Shortfall',
    'Deviation (MW)': 'Deviation',
    'Power Station': 'Power_Station',
}

STATE_ENTITIES = [
    'Andhra Pradesh', 'Andhra Pradesh.', 'Assam', 'Bihar', 'Chhatisgarh', 'Delhi', 'Gujarat',
    'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
    'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Odisha', 'Puducherry', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand',
    'West Bengal', 'Andaman & Nicobar Islands',
]


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw Excel export as-is (no cleaning)."""
    return pd.read_excel(path, sheet_name='Sheet1')


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Phase 2 cleaning pipeline to a raw dataframe."""
    df = df.rename(columns=RAW_COLUMN_MAP).copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    before = len(df)
    df = df[df['Power_Station'] != '0']
    df = df[~df['Power_Station'].isin(STATE_ENTITIES)]
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def load_and_clean(path: str) -> pd.DataFrame:
    return clean(load_raw(path))
