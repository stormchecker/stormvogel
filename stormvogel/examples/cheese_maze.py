from fractions import Fraction
from stormvogel import bird
from stormvogel.model import ModelType

# Observation IDs: encoded as the set of open directions (N/E/S/W), except for
# special states. Observations for corridor and bottom-row cells are exceptions
# to the adjacency rule (they get "NS", "cheese", or "dragon" regardless of
# their actual adjacency structure).
_OBS_ID = {
    "start": 0,
    "NS": 1,  # all blue corridor cells: only N and S are open
    "cheese": 2,
    "dragon": 3,
    "ES": 4,  # top-row left corner
    "EW": 5,  # top-row non-junction cells (above a wall)
    "ESW": 6,  # top-row junction cells (above a corridor)
    "SW": 7,  # top-row right corner
}


def create_cheese_maze(
    num_corridors: int = 3,
    vcorridor_length: int = 2,
    slippery: float = 0.0,
):
    """Build the cheese maze POMDP using the bird API.

    The maze has a horizontal corridor (top row) of width ``2*num_corridors-1``
    connected to ``num_corridors`` vertical corridors.  Each vertical corridor
    ends at the bottom row, where the middle corridor holds the cheese and the
    rest hold dragons.

    Observations are determined by which directions are open (walls / grid
    boundaries), except for the bottom row and the special start state.

    :param num_corridors: Number of vertical corridors. Must be odd and >= 3.
    :param vcorridor_length: Number of blue cells in each vertical corridor
        (rows between horizontal corridor and bottom row).
    :param slippery: Probability that an action is a no-op (agent stays in
        place). Must be in [0, 1). Default 0 gives fully deterministic moves.
        Has no effect when the intended move would already stay in place (wall
        or boundary), since both outcomes coincide.
    """
    if num_corridors % 2 != 1 or num_corridors < 3:
        raise ValueError("num_corridors must be odd and >= 3")
    if vcorridor_length < 1:
        raise ValueError("vcorridor_length must be >= 1")
    if not 0 <= slippery < 1:
        raise ValueError("slippery must be in [0, 1)")

    top_row_width = 2 * num_corridors - 1
    corridor_cols = list(range(0, top_row_width, 2))  # [0, 2, 4, ...]
    middle_col = corridor_cols[num_corridors // 2]
    bottom_row = vcorridor_length + 1

    def _is_valid(row: int, col: int) -> bool:
        if row == 0:
            return 0 <= col < top_row_width
        return 1 <= row <= bottom_row and col % 2 == 0 and 0 <= col < top_row_width

    def available_actions(s) -> list[str]:
        if getattr(s, "type", None) == "start":
            return ["start"]
        return ["north", "south", "east", "west"]

    def delta(s, action: str):
        if getattr(s, "type", None) == "start":
            return [
                (
                    Fraction(1, num_corridors),
                    bird.BirdState(row=vcorridor_length, col=c),
                )
                for c in corridor_cols
            ]
        row, col = s.row, s.col
        dr, dc = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}[
            action
        ]
        nr, nc = row + dr, col + dc
        if not _is_valid(nr, nc):
            nr, nc = row, col
        if slippery == 0 or (nr == row and nc == col):
            return [(1, bird.BirdState(row=nr, col=nc))]
        return [
            (slippery, bird.BirdState(row=row, col=col)),
            (1 - slippery, bird.BirdState(row=nr, col=nc)),
        ]

    def observations(s) -> int:
        if getattr(s, "type", None) == "start":
            return _OBS_ID["start"]
        row, col = s.row, s.col
        if row == bottom_row:
            return _OBS_ID["cheese"] if col == middle_col else _OBS_ID["dragon"]
        if row > 0:
            return _OBS_ID["NS"]
        # top row: observation = open directions
        has_s = col % 2 == 0
        has_e = col + 1 < top_row_width
        has_w = col > 0
        if has_s and has_e and has_w:
            return _OBS_ID["ESW"]
        if has_s and has_e:
            return _OBS_ID["ES"]
        if has_s and has_w:
            return _OBS_ID["SW"]
        return _OBS_ID["EW"]

    def labels(s) -> list[str] | None:
        if getattr(s, "type", None) == "start":
            return None
        row, col = s.row, s.col
        if row == bottom_row:
            return ["cheese"] if col == middle_col else ["dragon"]
        return None

    def friendly_names(s) -> str:
        if getattr(s, "type", None) == "start":
            return "start"
        return f"({s.row},{s.col})"

    model = bird.build_bird(
        delta=delta,
        init=bird.BirdState(type="start"),
        labels=labels,
        friendly_names=friendly_names,
        available_actions=available_actions,
        observations=observations,
        modeltype=ModelType.POMDP,
    )

    _obs_name = {str(v): k for k, v in _OBS_ID.items()}
    for obs in list(model.observation_aliases):
        raw = model.observation_aliases[obs]
        if raw in _obs_name:
            model.observation_aliases[obs] = _obs_name[raw]

    return model


if __name__ == "__main__":
    print(create_cheese_maze().to_dot())
