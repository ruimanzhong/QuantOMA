"""Gold options overlay utilities.

This package is intentionally side-car / optional.  It does not alter the
existing AU9999 spot/futures signal pipeline unless wired from
``market_quant.gold.live_pipeline``.
"""

from market_quant.options.live import build_live_option_plan

__all__ = ["build_live_option_plan"]
