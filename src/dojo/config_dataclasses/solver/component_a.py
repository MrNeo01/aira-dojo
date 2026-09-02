"""Configuration for the FAS Component-A solver."""

from dataclasses import dataclass, field

from dojo.config_dataclasses.solver.greedy import GreedySolverConfig


@dataclass
class ComponentASolverConfig(GreedySolverConfig):
    """Greedy mechanics plus a phase-aware FRR-UCB operator policy."""

    component_a_window_size: int = field(
        default=8,
        metadata={"description": "Common action/reward window size per phase."},
    )
    component_a_decay: float = field(
        default=0.5,
        metadata={"description": "Fitness-rate-rank decay D."},
    )
    component_a_exploration: float = field(
        default=0.25,
        metadata={"description": "UCB exploration coefficient alpha."},
    )
    component_a_epsilon: float = field(
        default=1e-12,
        metadata={"description": "Positive numerical and reward-cost floor."},
    )
    component_a_phase_boundaries: list[float] = field(
        default_factory=lambda: [1.0 / 3.0, 2.0 / 3.0],
        metadata={"description": "Consumed-budget boundaries for early/middle/late."},
    )
    component_a_seed: int = field(
        default=42,
        metadata={"description": "Seed for local policy and parent-selection RNGs."},
    )

    def validate(self) -> None:
        super().validate()
        if self.use_test_score:
            raise ValueError("Component A requires use_test_score=False")
        if self.step_limit < 2:
            raise ValueError("Component A requires at least one candidate evaluation")
        if self.num_drafts < 1:
            raise ValueError("num_drafts must be positive")
        if self.num_drafts >= self.step_limit - 1:
            raise ValueError(
                "Component A requires num_drafts < step_limit - 1 so at least "
                "one candidate evaluation is selected by the bandit"
            )
        if self.max_debug_depth < 0:
            raise ValueError("max_debug_depth must be non-negative")
        if self.component_a_window_size < 1:
            raise ValueError("component_a_window_size must be positive")
        if not 0.0 <= self.component_a_decay <= 1.0:
            raise ValueError("component_a_decay must be in [0, 1]")
        if self.component_a_exploration < 0.0:
            raise ValueError("component_a_exploration must be non-negative")
        if self.component_a_epsilon <= 0.0:
            raise ValueError("component_a_epsilon must be positive")
        if len(self.component_a_phase_boundaries) != 2:
            raise ValueError("component_a_phase_boundaries must have two entries")
        early_end, middle_end = self.component_a_phase_boundaries
        if not 0.0 < early_end < middle_end < 1.0:
            raise ValueError(
                "component_a_phase_boundaries must satisfy 0 < early < middle < 1"
            )
