# Breaking Changes

Below is a comprehensive list of breaking changes that affect users upgrading to this version.

## Module layout

The single `stormvogel/model.py` file has been split into a package `stormvogel/model/` with submodules (`model.py`, `state.py`, `action.py`, `choices.py`, `distribution.py`, `observation.py`, `reward_model.py`, `value.py`, `variable.py`). All public names are re-exported from `stormvogel.model`, so `import stormvogel.model` continues to work.


## `bird` builder

| Old | New |
|---|---|
| `rewards` accepted `(state, action)` for action models and `(state)` for non-action models | `rewards` must always accept exactly one parameter: `(state)` |
| `valuations` callback returned `dict[str, ...]` | `valuations` callback must return `dict[Variable, ...]` |


## `Model` API

### Attribute changes

| Old | New |
|---|---|
| `model.type` | `model.model_type` |
| `model.states: dict[int, State]` | `model.states: list[State]` |
| `model.choices: dict[int, Choices]` | `model.transitions: dict[State, Choices]` |
| `model.actions: set[Action] \| None` (attribute) | `model.actions` (property, yields from `transitions`) |
| `model.exit_rates: dict[int, ValueType] \| None` | Removed — rates are encoded in transition values |

### Getters replaced by properties

| Old | New |
|---|---|
| `model.get_initial_state()` | `model.initial_state` |
| `model.get_variables()` returned `set[str]` | `model.variables` returns `set[Variable]` |
| `model.get_parameters()` | `model.parameters` |

`model.get_labels()` (which returned `set[str]`) has been removed. The closest equivalent is `model.state_labels`, but note the type changed: it is a `dict[str, set[State]]` mapping each label to its states. Use `set(model.state_labels.keys())` to get the old behavior.

### Renames

| Old | New |
|---|---|
| `model.set_choice(state, choices)` | `model.set_choices(state, choices)` |
| `model.add_choice(state, choices)` | `model.add_choices(state, choices)` |
| `model.parameter_valuation(values)` | `model.get_instantiated_model(values)` |
| `choice_from_shorthand(shorthand, model)` | `choices_from_shorthand(shorthand)` (no `model` parameter) |

### Signature changes

| Old | New |
|---|---|
| `for id, state in model:` | `for state in model:` |
| `model[state_id]` (by int id) | `model[index]` (by list index) |
| `model.get_state_by_id(int)` | `model.get_state_by_id(UUID)` — for integer index access use `model.states[i]` |
| `model.get_states_with_label()` returned `list[State]` | Returns `set[State]` |
| `model.get_sub_model(states: list[State])` | `model.get_sub_model(states: Iterable[State])` |
| `model.iterate_transitions()` returned `list` | Returns `Iterator[tuple[ValueType, State]]` |
| `model.unassigned_variables()` returned `list` | Returns `Iterator[tuple[State, Variable]]` |
| `model.get_successor_states(state)` returned `set[int]` | Returns `set[State]` |
| `model.remove_state(state, normalize, reassign_ids)` | `model.remove_state(state, normalize, suppress_warning)` — `reassign_ids` removed |
| `model.make_observations_deterministic(reassign_ids)` | `model.make_observations_deterministic()` — `reassign_ids` removed |
| `model.observation(observation_id: int, valuations)` | `model.observation(alias: str)` — create-or-get by alias |
| `model.new_observation(valuations, observation_id)` | `model.new_observation(alias: str)` |
| `model.new_state(..., name=)` accepted a `name` argument | The `name` parameter has been removed |

### New methods

| Method | Notes |
|---|---|
| `model.get_state_index(state) -> int` | O(1) amortized index of a state in `model.states` |
| `model.get_distribution(state) -> Distribution` | Replaces `model.get_branches(state)` |
| `model.get_observation(alias: str) -> Observation` | Strict lookup; raises `KeyError` if not found |
| `model.new_reward_model(name: str) -> RewardModel` | Factory for reward models |
| `model.get_default_rewards() -> RewardModel` | Returns the first reward model |
| `model.get_rewards(name: str) -> RewardModel` | Looks up a reward model by name |

`ModelType.HMM` has been added alongside `POMDP`. Use `new_hmm()` to create an HMM model.

### Removals

The following methods have been removed without a direct replacement:

- `model.get_state_by_name()` — use labels to identify states.
- `model.get_action()` and `model.get_action_with_label()` — use `model.action("label_string")` instead.
- `model.get_choices(state)` and `model.get_choice(state)` — use `state.choices` instead.
- `model.get_states()` — use `model.states` directly.
- `model.get_ordered_labels()` — build from `model.state_labels` if needed.
- `model.get_rate()` and `model.set_rate()` — rates are now encoded in transition values.
- `model.get_choice_id()` and `model.get_state_action_id()`.
- `model.get_state_action_pair()`.
- `model.remove_transitions_between_states()`.
- `model.all_non_init_states_incoming_transition()`.
- `model.reassign_ids()`.
- `model.has_only_number_values()`.
- `Model.__eq__` — model comparison now uses identity (`is`), not structural equality.


## `State` API

### Identity and hashing

| Old | New |
|---|---|
| `state.id: int` | `state.state_id: UUID` |
| `@dataclass()` with custom `__eq__` comparing integer id, labels, valuations | `@dataclass(eq=False)` — identity-based equality |
| Unhashable (could not be used as dict key) | Hashable by identity (used as dict key throughout) |

### Attributes and properties

| Old | New |
|---|---|
| `state.labels` was a `list[str]` attribute | `state.labels` is a property returning `Iterable[str]` (backed by `model.state_labels`) |
| `state.get_choices()` / `state.get_choice()` | `state.choices` (property) |
| `state.get_observation()` / `state.set_observation()` | `state.observation` (property with setter) |
| `state.nr_choices()` (method) | `state.nr_choices` (property) |
| `state.add_valuation(variable: str, value)` | `state.add_valuation(variable: Variable, value)` |
| `state.get_valuation(variable: str)` | `state.get_valuation(variable: Variable)` |
| *(absent)* | `state.friendly_name -> str \| None` (property) |
| *(absent)* | `state.set_friendly_name(name)` |

### Renames

| Old | New |
|---|---|
| `state.set_choice(choices)` | `state.set_choices(choices)` |
| `state.add_choice(choices)` | `state.add_choices(choices)` |


## `Choices` API

The `Branches` class has been **removed**. Each action in a `Choices` now maps directly to a `Distribution[ValueType, State]` object.

| Old | New |
|---|---|
| `choices.choice: dict[Action, Branches]` (attribute) | Use `choices[action]` or `for action, branch in choices:` |
| `choices.actions()` (method) | `choices.actions` (property) |
| `Branches.branch: list[tuple[Value, State]]` | `Distribution[Value, State]` (see `Distribution` API below) |
| `Branches.get_successors()` returning `set[int]` | `Distribution.support` (property) returning `set[State]` |
| `Branches.sum_probabilities()` | Removed |
| `Choices.sum_probabilities(action)` | Removed |
| `Choices.__eq__`, `Branches.__eq__` (structural) | Removed — identity-based equality |


## `Distribution` API (new, replaces `Branches`)

`Branches` has been replaced by the generic `Distribution[ValueType, SupportType]` class. A `Choices` object now maps each `Action` directly to a `Distribution[ValueType, State]`.

| Feature | API |
|---|---|
| Construction from list | `Distribution([(prob, state), ...])` |
| Construction from dict | `Distribution({state: prob, ...})` |
| Iteration | `for value, state in distribution:` |
| Support set (replaces `get_successors()`) | `distribution.support` → `set[State]` |
| Probability list | `distribution.probabilities` → `list[ValueType]` |
| Stochasticity check | `distribution.is_stochastic(epsilon)` |


## `RewardModel` API

| Old | New |
|---|---|
| `RewardModel.rewards: dict[tuple[int, Action], ValueType]` | `RewardModel.rewards: dict[State, ValueType]` |
| `get_state_reward(state)` raised on action models | `get_state_reward(state)` works on all models |
| `set_state_action_reward(state, action, value)` | Removed. Use `set_state_reward(state, value)`. |
| `get_state_action_reward(state, action)` | Removed. Use `get_state_reward(state)`. |
| `set_from_rewards_vector(vector)` | `set_from_rewards_vector(vector, state_action=False)`. When `state_action=True`, the vector is interpreted as one entry per (state, action) pair. |
| `RewardModel.__eq__` (structural) | Removed — identity-based equality |


## `Observation` API

| Old | New |
|---|---|
| `Observation.observation: int` | `Observation.alias: str` with a UUID `observation_id` |


## `Interval` fields

| Old | New |
|---|---|
| `Interval.bottom` | `Interval.lower` |
| `Interval.top` | `Interval.upper` |


## `Variable` type

State valuations previously used plain `str` keys; they now use `Variable` objects (a frozen dataclass with a `label: str` field, e.g. `Variable("x")`). All valuation-related APIs in `State` and `Model` are affected — see the respective sections above.
