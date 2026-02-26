# Breaking Changes

The codebase has undergone a major refactoring to improve type safety, code clarity, and consistency. Below is a list of breaking changes that affect users upgrading to this version:

## `Model` API
- The `Model` iterable now yields `State` objects directly instead of `(id, state)` tuples. Instead of `for _, state in model:`, use `for state in model:`.
- `model.get_initial_state()` has been replaced by the property `model.initial_state`.
- `model.get_variables()` has been replaced by the property `model.variables`.
- `model.type` has been renamed to `model.model_type`.
- `model.set_choice()` has been renamed to `model.set_choices()`.
- `model.get_state_by_id()` now expects a UUID instead of an integer ID. If you need to access a state by its numerical index, use `model.get_state_by_stormpy_id(id)` or `model.states[id]`.
- `model.get_states_with_label()` now returns a `set` instead of a `list`. Ensure you convert it to a `list` or use an iterator if you relied on subscripting (e.g., `list(model.get_states_with_label("A"))[0]`).
- `model.get_sub_model()` now requires its `states` parameter to be an `Iterable[State]` instead of `list[State]`.
- `model.iterate_transitions()` now returns an `Iterator[tuple[ValueType, State]]` instead of returning a `list`.
- `model.unassigned_variables()` now yields variables as an `Iterator[tuple[State, str]]` instead of returning a `list`.
- The state `name` mechanism has been entirely removed from the API. States must be identified using their `labels`. `model.get_state_by_name()` is removed, and `model.new_state` no longer accepts a `name=` argument.
- `model.get_action()` and `model.get_action_with_label()` have been removed. To retrieve or declare an action, directly use `model.action("label_string")`.
- `stormvogel.model.model.choice_from_shorthand` has been renamed to `stormvogel.model.model.choices_from_shorthand`.
## `RewardModel` API
- `RewardModel` rewards now only require a `state` and a `value` for their rewards, ignoring actions.
- The `set_state_action_reward(state, action, reward)` method has been removed. Use `set_state_reward(state, reward)` instead.

## `bird` module
- The `rewards` function provided to `bird.build_bird` must accept only one parameter: `(state)`. The previous expectation to provide a `(state, action)` function when the model supports actions has been completely removed to match the updated `RewardModel` API.
