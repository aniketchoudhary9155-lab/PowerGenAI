"""
PowerGenAI - Analytics Module (Phase 11 & 12 logic)
---------------------------------------------------------
Station-level performance monitoring and maintenance analytics, computed
using ratio-of-sums (not mean-of-ratios) for percentage metrics to avoid
distortion from near-zero-Programme records (see Phase 11 write-up).
"""

import numpy as np
import pandas as pd


def station_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Power_Station with robust aggregate performance metrics."""

    def _agg(g):
        prog_sum = g['Programme'].sum()
        cap_sum = g['Available_Capacity'].sum()
        mon_sum = g['Monitored_Capacity'].sum()
        return pd.Series({
            'Records': len(g),
            'Programme_avg': g['Programme'].mean(),
            'Actual_avg': g['Actual'].mean(),
            'Programme_sum': prog_sum,
            'Actual_sum': g['Actual'].sum(),
            'Generation_Difference_avg': g['Generation_Difference'].mean(),
            'Excess_Shortfall_avg': g['Excess_Shortfall'].mean(),
            'Total_Maintenance_avg': g['Total_Maintenance'].mean(),
            'Forced_Maintenance_avg': g['Forced_Maintenance'].mean(),
            'Programme_Achievement_pct': (g['Actual'].sum() / prog_sum * 100) if prog_sum > 0 else np.nan,
            'Capacity_Utilization_pct': (g['Actual'].sum() / cap_sum * 100) if cap_sum > 0 else np.nan,
            'Deviation_pct': ((g['Actual'].sum() - prog_sum) / prog_sum * 100) if prog_sum > 0 else np.nan,
            'Maintenance_Impact_Index_pct': (g['Total_Maintenance'].sum() / mon_sum * 100) if mon_sum > 0 else np.nan,
        })

    return df.groupby('Power_Station').apply(_agg, include_groups=False).reset_index()


def rankings(station_perf: pd.DataFrame, min_records: int = 50) -> dict:
    """Return the standard set of Phase 11 rankings, using only stations
    with at least `min_records` for statistical reliability."""
    reliable = station_perf[station_perf['Records'] >= min_records].copy()
    return {
        'best_performing': reliable.sort_values('Programme_Achievement_pct', ascending=False).head(10),
        'lowest_performing': reliable.sort_values('Programme_Achievement_pct', ascending=True).head(10),
        'highest_maintenance': reliable.sort_values('Total_Maintenance_avg', ascending=False).head(10),
        'highest_forced_maintenance': reliable.sort_values('Forced_Maintenance_avg', ascending=False).head(10),
        'highest_deviation': reliable.reindex(
            reliable['Deviation_pct'].abs().sort_values(ascending=False).index
        ).head(10),
        'highest_shortfall': reliable.sort_values('Excess_Shortfall_avg', ascending=True).head(10),
    }


def maintenance_composition(df: pd.DataFrame) -> dict:
    """Fleet-wide breakdown of maintenance by cause."""
    planned = df['Planned_Maintenance'].sum()
    forced = df['Forced_Maintenance'].sum()
    other = df['Other_Reasons'].sum()
    total = planned + forced + other
    if total == 0:
        return {'Planned': 0, 'Forced': 0, 'Other': 0}
    return {
        'Planned': planned / total * 100,
        'Forced': forced / total * 100,
        'Other': other / total * 100,
    }


def total_monitored_capacity(df: pd.DataFrame) -> float:
    """Correct fleet/station-level total capacity.

    IMPORTANT: Monitored_Capacity is a static attribute of a physical
    generating unit, but the dataset has many repeated observation-rows per
    unit (different time-blocks, same capacity figure). Naively summing
    Monitored_Capacity across all rows overcounts massively (verified 590x
    on this dataset - 284M MW vs the correct ~481,000 MW). The correct total
    is the sum of DISTINCT (Power_Station, Monitored_Capacity) combinations,
    which approximates one entry per physical sub-unit.

    Programme/Actual do NOT have this problem — they vary row-to-row
    (confirmed: 94 unique Programme values and 661 unique Actual values
    across 679 rows sharing one capacity figure), so summing those across
    all rows is legitimate and represents real distinct generation.
    """
    unique_capacity = df[['Power_Station', 'Monitored_Capacity']].drop_duplicates()
    return unique_capacity['Monitored_Capacity'].sum()
