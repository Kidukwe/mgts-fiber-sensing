"""Versioned column order shared by model fitting, artifacts and inference."""

FEATURE_NAMES = ["last_change_median", "last_change_hot", "last_change_cold",
                 "older_change_median", "older_change_hot", "older_change_cold",
                 "level_from_history_median", "level_from_history_hot", "level_from_history_cold",
                 "hot", "cold", "std", "roughness", "gradient", "high_frequency_fraction",
                 "robust_hot", "robust_cold", "fraction_hot", "fraction_cold",
                 "longest_hot_fraction", "longest_cold_fraction", "pair_fraction",
                 "sin_local_hour", "cos_local_hour"]
TARGET_NAMES = ["next_median_change_c", "next_hot_excursion_change_c",
                "next_cold_excursion_change_c"]
