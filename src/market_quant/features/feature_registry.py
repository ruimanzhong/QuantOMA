"""Feature builder registry for Phase 1."""

from __future__ import annotations

from market_quant.features.alpha_style_fallback import build_alpha_style_fallback_features
from market_quant.features.availability import apply_feature_availability_lag
from market_quant.features.qlib_alpha158_adapter import build_qlib_alpha158_features


FEATURE_BUILDERS = {
    "qlib_alpha158": build_qlib_alpha158_features,
    "alpha_style_fallback": build_alpha_style_fallback_features,
}

FEATURE_UTILS = {
    "apply_feature_availability_lag": apply_feature_availability_lag,
}
