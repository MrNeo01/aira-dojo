"""FAS Component A: online phase-aware operator allocation."""

from dojo.components.component_a.bandit import (
    ARMS,
    PHASES,
    BanditDecision,
    BanditSnapshot,
    PhaseAwareFRRUCBPolicy,
    budget_phase,
    incumbent_progress_reward,
    require_validation_only,
)

__all__ = [
    "ARMS",
    "PHASES",
    "BanditDecision",
    "BanditSnapshot",
    "PhaseAwareFRRUCBPolicy",
    "budget_phase",
    "incumbent_progress_reward",
    "require_validation_only",
]
