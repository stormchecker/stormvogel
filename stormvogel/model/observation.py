from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Any


@dataclass(eq=False)
class Observation:
    """Represents an observation of a state (for POMDPs and HMMs)

    Args:
        observation: the observation of a state as an integer
    """

    alias: str
    observation_id: UUID = field(default_factory=uuid4)
    valuations: dict[str, Any] | None = None

    def format(self):
        """Formats the observation for visualizations."""
        return f"{self.alias} {self.valuations if self.valuations is not None else ''}"

    def __str__(self):
        return f"Obs: {self.alias} ({self.observation_id})"
