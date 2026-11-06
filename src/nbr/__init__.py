"""Hierarchical next-basket recommendation (Instacart, Complete Journey).

Core: config, data (loading with holdout guard), io (JSON with provenance), metrics (per-user
metrics, paired bootstrap). Lists: baselines, repeat (R1), explore (E1, E4), itemcf (E3),
usercf (E2), features + lr_repeat (R2), features_cycle (R3 screening), combine (slot merge).
Orchestration: pipeline (tune / evaluate), __main__ (CLI); stand-alone screening runners
usercf_eval (W7) and r3_eval (W9).
"""

__version__ = "0.1.0"
