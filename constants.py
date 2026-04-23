"""constants.py
--------------
Shared time-grid constants for all stochastic-process modules.

Convention: 1 trading year = 252 days × 390 minutes per day.
"""

_MINS_PER_DAY   = 390                            # minutes per NYSE trading day
_DT_MIN         = 1.0 / (252.0 * 390.0)         # 1 minute in year-fraction
_OVERNIGHT_MINS = 1050.0                          # nominal overnight gap in minutes
_DT_OVERNIGHT   = (_OVERNIGHT_MINS + 1) / (252.0 * 390.0)
_DT_OVERWEEKEND = (2 * 60 * 24 + _OVERNIGHT_MINS + 1) / (252.0 * 390.0)