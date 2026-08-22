"""
PowerGenAI - Utility functions
-------------------------------------
Shared formatting and input-validation helpers used across the dashboard.
"""


def fmt_mw(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{value:,.1f} MW"
    except (TypeError, ValueError):
        return "N/A"


def fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{value:,.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def validate_prediction_inputs(monitored_capacity, programme, planned_maintenance,
                                 forced_maintenance, other_reasons) -> list:
    """Returns a list of human-readable error messages (empty if all valid).
    Covers the Phase 14 error-handling requirements: missing input, invalid
    numeric input, negative capacity, zero capacity/programme."""
    errors = []
    fields = {
        'Monitored Capacity': monitored_capacity,
        'Programme': programme,
        'Planned Maintenance': planned_maintenance,
        'Forced Maintenance': forced_maintenance,
        'Other Reasons': other_reasons,
    }
    for name, val in fields.items():
        if val is None:
            errors.append(f"{name} is required.")
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"{name} must be a valid number.")
            continue
        if val < 0:
            errors.append(f"{name} cannot be negative.")

    if not errors:
        if monitored_capacity == 0:
            errors.append("Monitored Capacity is zero — prediction would be meaningless for a plant with no capacity.")
        total_maint = (planned_maintenance or 0) + (forced_maintenance or 0) + (other_reasons or 0)
        if monitored_capacity and total_maint > monitored_capacity:
            errors.append(
                f"Total maintenance ({total_maint:.1f} MW) exceeds Monitored Capacity "
                f"({monitored_capacity:.1f} MW) — please check your inputs."
            )
        if programme == 0:
            errors.append("Warning: Programme is zero — Programme Achievement % cannot be calculated.")

    return errors
