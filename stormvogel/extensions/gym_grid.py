from stormvogel import bird
from typing import Callable
import stormvogel.result
import stormvogel.model
import math
import imageio
import os

GRID_ACTION_LABEL_MAP = {0: "←", 1: "↓", 2: "→", 3: "↑", 4: "pickup", 5: "dropoff"}


def gymnasium_grid_to_stormvogel(
    env, action_label_map: dict[int, str] = GRID_ACTION_LABEL_MAP
):
    """Convert a FrozenLake, Taxi, or Cliffwalking gymnasium environment to an explicit stormvogel model."""
    TRANSITIONS = env.unwrapped.P
    NO_ACTIONS = env.action_space.n
    INV_MAP = {v: k for k, v in action_label_map.items()}
    ALL_ACTIONS = [[x] for x in action_label_map.values()]

    def action_numer_map(a: bird.Action):
        return INV_MAP[a[0]]

    if "taxi" in env.spec.id.lower():
        # For Taxi, we need a special initial state that goes to every state. This is to account for the randomized starting position.
        init = bird.State(n=-1, done=False)
    else:
        init = bird.State(
            n=0, done=False
        )  # Otherwise, it's just 0 (Cliffwalking and FrozenLake).

    def available_actions(s: bird.State):
        if s.n == -1:
            return [[]]
        return ALL_ACTIONS[:NO_ACTIONS]

    def delta(s: bird.State, a: bird.Action):
        if (
            s.n == -1
        ):  # Special taxi init state. It goes to every location that a passenger could spawn in. This should explore all states.
            # state = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
            PLS = 4  # Number of passenger locations.
            return [(1 / PLS, bird.State(n=x, done=False)) for x in range(PLS)]
        trans = TRANSITIONS[s.n][action_numer_map(a)]
        return list(map(lambda x: (x[0], bird.State(n=int(x[1]), done=x[3])), trans))

    def rewards(s: bird.State, a: bird.Action) -> dict[str, stormvogel.model.Value]:
        if s.n == -1:
            return {"R": 0}
        reward = list(map(lambda x: x[2], TRANSITIONS[s.n][action_numer_map(a)]))[0]
        return {"R": reward}

    def labels(s: bird.State):
        labels = [str(s.n), str(to_coordinate(s.n, env))]
        if s.n == get_target_state(env):
            labels.append("target")
        if s.done:
            labels.append("done")
        # labels.append("always")
        return labels

    return bird.build_bird(
        delta=delta,
        init=init,
        available_actions=available_actions,
        labels=labels,
        rewards=rewards,
        modeltype=stormvogel.model.ModelType.MDP,
    )


def to_coordinate(s, env):
    """Calculate the state's coordinates. Works for FrozenLake, Cliffwalking, and Taxi"""
    num_states = env.observation_space.n
    grid_size = int(math.sqrt(num_states))
    x_target = int(s % grid_size)
    y_target = int(s // grid_size)
    return x_target, y_target


def to_state(x, y, env):
    """Calculate the state index from coordinates. Works for FrozenLake, CliffWalking, and Taxi."""
    num_states = env.observation_space.n
    grid_size = int(math.sqrt(num_states))
    return y * grid_size + x


def get_target_state(env):
    """Calculate the target state for an env. Works for FrozenLake and Cliffwalking"""
    return env.observation_space.n - 1


def to_gymnasium_scheduler(
    model: stormvogel.model.Model,
    scheduler: stormvogel.result.Scheduler
    | Callable[[stormvogel.model.State], stormvogel.model.Action],
    action_label_map: dict[int, str] = GRID_ACTION_LABEL_MAP,
) -> Callable[[int], int]:
    """Convert a stormvogel scheduler to a gymnasium scheduler (for a model that was converted using gymnasium_grid_to_stormvogel).
    Args:
        model: Stormvogel model.
        scheduler: Stormvogel scheduler.
        action_label_map: Map that you also used for the call in gymnasium_grid_to_stormvogel.
    Returns a gymnasium scheduler."""
    inv_map = {v: k for k, v in action_label_map.items()}

    def gymnasium_scheduler(env_sid: int):
        # TODO change this once bird API features are a thing.
        model_state = model.get_states_with_label(str(int(env_sid)))[0]
        if isinstance(scheduler, stormvogel.result.Scheduler):
            choice = scheduler.get_choice_of_state(model_state)
        elif callable(scheduler):
            choice = scheduler(model_state)  # type: ignore
        return inv_map[list(choice.labels)[0]]

    return gymnasium_scheduler


def gymnasium_render_model_gif(
    env,
    gymnasium_scheduler: Callable[[int], int] | None = None,
    filename: str = "my_gif",
    max_length: int = 50,
    fps: int = 2,
    loop: int = 0,
) -> str:
    """Render a gymnasium model to a gif, using the gymnasium_scheduler (A map from state numbers to action numbers) to pick an action.
    Leave as None for a random action.

    Args:
        env: The gymnasium environment.
        gymnasium_scheduler: A function that takes a state number and returns an action number.
        filename: The name of the gif file to save.
        max_length: The maximum number of frames to render.
        fps: Frames per second for the gif.
        loop: Number of times to loop the gif (0 means loop forever).
    """
    frames = []  # List to store frames

    state, info = env.reset()
    for _ in range(max_length):
        frame = env.render()
        frames.append(frame)  # Store the frame

        action = (
            env.action_space.sample()
            if gymnasium_scheduler is None
            else gymnasium_scheduler(state)
        )
        state, reward, done, truncated, info = env.step(action)

        if done:
            frame = env.render()
            frames.append(frame)  # Render the last frame
            break  # Stop if the episode ends

    env.close()
    os.makedirs("gifs", exist_ok=True)
    # Save frames as a GIF
    imageio.mimsave(
        "gifs/" + filename + ".gif", frames, fps=fps, loop=loop
    )  # Adjust FPS as needed
    return "gifs/" + filename + ".gif"
