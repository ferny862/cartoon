"""Display names and fixed colors for strategies (color follows the entity)."""

DISPLAY_NAMES = {
    "benchmark": "SPY buy and hold",
    "trend_faber": "Trend filter (Faber)",
    "gem": "GEM",
    "gtaa5": "GTAA (5 assets)",
    "factor_blend": "Factor blend",
    "sector_mom_sector_filter": "Sector momentum + sector filter",
    "sector_mom_market_filter": "Sector momentum + market filter",
}

# Categorical slot per strategy, in config order. The benchmark is drawn in
# neutral ink, never a categorical hue.
SLOT = {
    "trend_faber": 0,
    "gem": 1,
    "gtaa5": 2,
    "factor_blend": 3,
    "sector_mom_sector_filter": 4,
    "sector_mom_market_filter": 5,
}


def display_name(name: str) -> str:
    base = name.replace("__french", "")
    label = DISPLAY_NAMES.get(base, base)
    return f"{label} (French industries)" if name.endswith("__french") else label
