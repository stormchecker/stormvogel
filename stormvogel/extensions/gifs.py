"""Create gifs from stormvogel models."""

import stormvogel.simulator as simulator
import stormvogel.model
from typing import Callable
import imageio
from PIL.Image import Image
import os


def render_model_gif(
    model: stormvogel.model.Model,
    state_to_image: Callable[[stormvogel.model.State], Image],
    scheduler: (
        stormvogel.result.Scheduler
        | Callable[[stormvogel.model.State], stormvogel.model.Action]
        | None
    ) = None,
    path: simulator.Path | None = None,
    filename: str = "my_gif",
    max_length: int = 50,
    fps: int = 2,
    loop: int = 0,
) -> str:
    """Render a stormvogel model to a gif.

    Use *scheduler* to pick actions; leave as ``None`` for random actions.

    :param model: Stormvogel model.
    :param state_to_image: Function that takes a state and returns an image.
    :param scheduler: Scheduler to use for the simulation.
    :param path: Path to use for the simulation. If both *scheduler* and *path*
        are set, *path* takes precedence.
    :param filename: The gif will be saved as ``gifs/<filename>.gif``.
    :param max_length: Maximum number of steps to simulate.
    :param fps: Frames per second of the gif.
    :param loop: Number of times the gif loops. ``0`` means infinite.
    :returns: Filesystem path of the saved gif.
    """
    if path is None:
        if model.supports_actions() and scheduler is not None:
            path = simulator.simulate_path(model, max_length, scheduler)
        else:
            path = simulator.simulate_path(model, max_length)
    frames = [state_to_image(model.initial_state)]  # List to store frames

    for i in range(1, min(max_length, len(path))):
        state = path.get_step(i)
        if not isinstance(state, stormvogel.model.State):
            state = state[1]
        frames.append(state_to_image(state))

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
    """Embed a gif in a Jupyter notebook so that it also works with Sphinx docs.

    :param filename: Path to the gif file.
    """
    import IPython.display as ipd

    with open("GIF" + ".html", "w") as f:
        f.write(f'<img src="{filename}">')
    ipd.display(ipd.HTML(filename="GIF" + ".html"))
