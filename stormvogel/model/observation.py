from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Any, TYPE_CHECKING

from stormvogel.model.variable import Variable

if TYPE_CHECKING:
    from stormvogel.model import Model


@dataclass(order=False, eq=False, frozen=True)
class Observation:
    """Represent an observation of a state (for POMDPs and HMMs).

    :param alias: Human-readable name for the observation.
    :param observation_id: Unique identifier for the observation.
    :param valuations: Optional mapping of variable names to observed values.
    """

    model: "Model"
    observation_id: UUID = field(default_factory=uuid4)

    @property
    def valuations(self) -> dict[Variable, Any]:
        """Return the variable-value pairs observed in this observation."""
        if self not in self.model.observation_valuations:
            raise RuntimeError(
                f"Observation {self.observation_id} does not have valuations in the model."
            )
        return self.model.observation_valuations[self]

    @property
    def alias(self) -> str:
        """Return the alias of this observation."""
        if self not in self.model.observation_aliases:
            raise RuntimeError(
                f"Observation {self.observation_id} does not have an alias in the model."
            )
        return self.model.observation_aliases[self]

    def display(self):
        """Format the observation for visualizations."""
        if not self.valuations:
            return self.alias
        return f"{self.alias} {self.valuations}"

    def __str__(self):
        return f"Obs: {self.alias} ({self.observation_id})"
