from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Any


@dataclass(eq=False)
class Observation:
    """Represent an observation of a state (for POMDPs and HMMs).

    :param alias: Human-readable name for the observation.
    :param observation_id: Unique identifier for the observation.
    :param valuations: Optional mapping of variable names to observed values.
    """

    alias: str
    observation_id: UUID = field(default_factory=uuid4)
    valuations: dict[str, Any] | None = None

    def format(self):
        """Format the observation for visualizations."""
        return f"{self.alias} {self.valuations if self.valuations is not None else ''}"

    def __str__(self):
        return f"Obs: {self.alias} ({self.observation_id})"
