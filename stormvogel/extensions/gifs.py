import stormvogel.simulator as simulator
import stormvogel.model
from typing import Callable
import imageio
from PIL.Image import Image
import os


def render_model_gif(
    model: stormvogel.model.Model,
    state_to_image: Callable[[stormvogel.model.State], Image],
    scheduler: stormvogel.result.Scheduler
    | Callable[[stormvogel.model.State], stormvogel.model.Action]
    | None = None,
    path: simulator.Path | None = None,
    filename: str = "my_gif",
    max_length: int = 50,
    fps: int = 2,
    loop: int = 0,
) -> str:
    """Render a stormvogel model to a gif, using the gymnasium_scheduler (A map from state numbers to action numbers) to pick an action.
    Leave as None for a random action.

    Args:
        model (Model): stormvogel model.
        state_to_image (Callable[[State], Image]): Function that takes a state and returns an image.
        scheduler (Scheduler | Callable[[State], Action] | None): Scheduler to use for the simulation.
        path (Path | None): Path to use for the simulation. If both scheduler and path are set, path takes presedence.
        filename (str): The gif will be saved as gifs/filename.gif.
        max_length (int): The maximum amount of steps to simulate.
        fps (int): The frames per second of the gif.
        loop (int): The amount of times the gif will loop. 0 means infinite."""
    if path is None:
        if model.supports_actions() and scheduler is not None:
            path = simulator.simulate_path(model, max_length, scheduler)
        else:
            path = simulator.simulate_path(model, max_length)
    frames = [state_to_image(model.get_initial_state())]  # List to store frames

    for i in range(1, min(max_length, len(path) + 1)):
        state = path.get_step(i)
        if not isinstance(state, stormvogel.model.State):
            state = state[1]
        frames.append(state_to_image(state))  # type: ignore

    os.makedirs("gifs", exist_ok=True)
    # Save frames as a GIF
    imageio.mimsave(
        "gifs/" + filename + ".gif",
        frames,  # type: ignore
        fps=fps,
        loop=loop,
    )  # type: ignore
    return "gifs/" + filename + ".gif"


def embed_gif(filename: str):
    """Hacky way to embed a gif in a jupyter notebook so that it also works with sphinx docs."""
    import IPython.display as ipd

    with open("GIF" + ".html", "w") as f:
        f.write(f'<img src="{filename}">')
    ipd.display(ipd.HTML(filename="GIF" + ".html"))
