"""Validity checking for stormvogel models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.model.state import State


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    message: str
    context: Any = None

    def __str__(self) -> str:
        suffix = f" (context: {self.context})" if self.context is not None else ""
        return f"[{self.severity.value}] {self.message}{suffix}"


class ValidationResult:
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues

    @property
    def is_valid(self) -> bool:
        return all(i.severity != Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def __str__(self) -> str:
        if not self.issues:
            return "Model is valid."
        lines = [str(i) for i in self.issues]
        status = "valid (with warnings)" if self.is_valid else "invalid"
        return f"Model is {status}:\n" + "\n".join(lines)

    def __repr__(self) -> str:
        return f"ValidationResult(issues={self.issues!r})"


def _check_initial_state(model: "Model") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    init_states = model.state_labels.get("init", set())
    if len(init_states) != 1:
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                f"Model must have exactly one 'init' state, found {len(init_states)}.",
            )
        )
    return issues


def _check_transition_targets(model: "Model") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    state_set = set(model.states)
    for state, choices in model.transitions.items():
        for action, branch in choices:
            for _prob, target in branch:
                if target not in state_set:
                    issues.append(
                        ValidationIssue(
                            Severity.ERROR,
                            f"Transition from {state!r} via {action!r} targets a state not in the model.",
                            context=target,
                        )
                    )
    return issues


def _check_distributions(model: "Model") -> list[ValidationIssue]:
    """Check that every distribution sums to 1 (skipped for parametric/interval models)."""
    issues: list[ValidationIssue] = []
    if model.is_parametric() or model.is_interval_model():
        return issues
    for state, choices in model.transitions.items():
        for action, branch in choices:
            if not branch.is_stochastic():
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        f"Distribution at state {state!r} for action {action!r} does not sum to 1.",
                        context=(state, action),
                    )
                )
    return issues


def _check_zero_transitions(model: "Model") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for state, choices in model.transitions.items():
        if choices.has_zero_transition():
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"State {state!r} has a transition with probability zero.",
                    context=state,
                )
            )
    return issues


def _check_deadlocks(model: "Model") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for state in model.states:
        if not state.has_choices():
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"State {state!r} has no outgoing choices (deadlock).",
                    context=state,
                )
            )
    return issues


def _check_reachability(model: "Model") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    init_states = model.state_labels.get("init", set())
    if len(init_states) != 1:
        return issues  # initial-state check already covers this
    init = next(iter(init_states))

    visited: set["State"] = set()
    queue: deque["State"] = deque([init])
    while queue:
        s = queue.popleft()
        if s in visited:
            continue
        visited.add(s)
        if s in model.transitions:
            for _action, branch in model.transitions[s]:
                for _prob, target in branch:
                    if target not in visited:
                        queue.append(target)

    unreachable = [s for s in model.states if s not in visited]
    for state in unreachable:
        issues.append(
            ValidationIssue(
                Severity.WARNING,
                f"State {state!r} is not reachable from the initial state.",
                context=state,
            )
        )
    return issues


def validate_shared(model: "Model") -> list[ValidationIssue]:
    """Run checks that apply to all model types."""
    issues: list[ValidationIssue] = []
    issues += _check_initial_state(model)
    issues += _check_transition_targets(model)
    issues += _check_distributions(model)
    issues += _check_zero_transitions(model)
    return issues


def validate_mdp(model: "Model") -> list[ValidationIssue]:
    """Run MDP-specific checks (deadlocks and reachability)."""
    issues: list[ValidationIssue] = []
    issues += _check_deadlocks(model)
    issues += _check_reachability(model)
    return issues


def validate(model: "Model") -> ValidationResult:
    """Validate a model and return a :class:`ValidationResult`.

    Shared checks (target states, stochasticity, initial state, zero transitions)
    are run for every model type.  MDP-specific checks (deadlocks, reachability)
    are added for MDP models.
    """
    from stormvogel.model.model import ModelType

    issues = validate_shared(model)
    if model.model_type == ModelType.MDP:
        issues += validate_mdp(model)
    return ValidationResult(issues)
