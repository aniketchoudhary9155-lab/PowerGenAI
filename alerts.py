"""
PowerGenAI - Alert System (Phase 13)
--------------------------------------
Configurable, rule-based alerting for power station performance.

IMPORTANT: All thresholds below are SOFTWARE-DEFINED ANALYTICAL THRESHOLDS
chosen for this project's demonstration purposes. They are NOT official
grid-code, CEA, or regulatory limits. Operators should replace them with
values from their own operating standards before production use.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class AlertConfig:
    """All thresholds are configurable. Defaults chosen from this dataset's
    own distribution (see Phase 4/11/12 EDA) purely for demonstration."""

    # Thresholds below are calibrated to roughly the 90th percentile (10th for
    # achievement) of THIS dataset's own distribution, so alerts flag the
    # worst ~5-10% of conditions rather than nearly half of all records.
    # Recalibrate against your own data before production use.

    # HIGH SHORTFALL: generation meaningfully below programme
    shortfall_mw_threshold: float = -12.0         # Generation_Difference below this (MW) [~5th pct]
    shortfall_pct_threshold: float = 58.5         # Programme_Achievement below this (%) [~10th pct]

    # HIGH MAINTENANCE: large share of capacity offline
    maintenance_impact_threshold: float = 50.0    # Maintenance_Impact_Index above this (%) [~90th pct]

    # HIGH FORCED MAINTENANCE: unplanned outages unusually high
    forced_maintenance_ratio_threshold: float = 0.50  # Forced_Maintenance / Monitored_Capacity above this [~90th pct]

    # HIGH DEVIATION: schedule volatility
    deviation_pct_threshold: float = 78.0         # |Deviation| above this (%) [~90th pct]


SEVERITY_MULTIPLIERS = {"MEDIUM": 1.0, "HIGH": 1.5, "CRITICAL": 2.0}


def _severity(value: float, threshold: float, higher_is_worse: bool = True) -> str:
    """Simple magnitude-based severity: how far past the threshold."""
    ratio = (value / threshold) if higher_is_worse else (threshold / value if value != 0 else np.inf)
    if abs(ratio) >= 2.0:
        return "CRITICAL"
    elif abs(ratio) >= 1.5:
        return "HIGH"
    else:
        return "MEDIUM"


def evaluate_row(row: pd.Series, config: AlertConfig) -> list[dict]:
    """Evaluate a single operational record against all configured rules.
    Returns a list of triggered alert dicts (empty if none triggered)."""
    alerts = []

    # HIGH SHORTFALL
    gen_diff = row.get("Generation_Difference", np.nan)
    prog_ach = row.get("Programme_Achievement", np.nan)
    if pd.notna(gen_diff) and gen_diff <= config.shortfall_mw_threshold:
        alerts.append({
            "type": "HIGH_SHORTFALL",
            "message": f"Generation {abs(gen_diff):.1f} MW below programme",
            "severity": _severity(gen_diff, config.shortfall_mw_threshold),
        })
    elif pd.notna(prog_ach) and prog_ach < config.shortfall_pct_threshold:
        alerts.append({
            "type": "HIGH_SHORTFALL",
            "message": f"Programme achievement only {prog_ach:.1f}%",
            "severity": _severity(prog_ach, config.shortfall_pct_threshold, higher_is_worse=False),
        })

    # HIGH MAINTENANCE
    mii = row.get("Maintenance_Impact_Index", np.nan)
    if pd.notna(mii) and mii >= config.maintenance_impact_threshold:
        alerts.append({
            "type": "HIGH_MAINTENANCE",
            "message": f"Maintenance Impact Index {mii:.1f}% of capacity",
            "severity": _severity(mii, config.maintenance_impact_threshold),
        })

    # HIGH FORCED MAINTENANCE
    fmr = row.get("Forced_Maintenance_Ratio", np.nan)
    if pd.notna(fmr) and fmr >= config.forced_maintenance_ratio_threshold:
        alerts.append({
            "type": "HIGH_FORCED_MAINTENANCE",
            "message": f"Forced maintenance is {fmr*100:.1f}% of monitored capacity",
            "severity": _severity(fmr, config.forced_maintenance_ratio_threshold),
        })

    # HIGH DEVIATION
    dev = row.get("Deviation", np.nan)
    if pd.notna(dev) and abs(dev) >= config.deviation_pct_threshold:
        alerts.append({
            "type": "HIGH_DEVIATION",
            "message": f"Deviation of {dev:.1f}% from schedule",
            "severity": _severity(abs(dev), config.deviation_pct_threshold),
        })

    return alerts


def evaluate_dataframe(df: pd.DataFrame, config: AlertConfig = None) -> pd.DataFrame:
    """Evaluate every row in a dataframe and return a long-format alerts table."""
    config = config or AlertConfig()
    records = []
    for idx, row in df.iterrows():
        for alert in evaluate_row(row, config):
            records.append({
                "row_index": idx,
                "Power_Station": row.get("Power_Station"),
                **alert,
            })
    return pd.DataFrame(records)
