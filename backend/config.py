"""
config.py
---------
Single location for tunable ShopBot configuration constants.
"""

# Search acceptance: a result set is accepted when its best score reaches
# SEARCH_FLOOR; products within SEARCH_MARGIN of that best score are kept.
# Chosen by scripts/measure_search.py on eval/nl_search_queries.csv.
SEARCH_FLOOR = 0.51
SEARCH_MARGIN = 0.16

# Weak results at or above this score may be shown, explicitly flagged as
# approximate. Below it the search returns no-match plus category suggestions.
SEARCH_APPROX_FLOOR = 0.38
