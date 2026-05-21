from pathlib import Path

from umbi.ats.ats_to_umb import write as _write_umb, read as _read_umb

from stormvogel.umbi.translate import translate_to_umbi, translate_to_stormvogel  # noqa: F401
from stormvogel.model.model import Model


def write_to_umb(
    model: Model,
    path: str | Path,
    ignore_unsupported_rewards: bool = False,
) -> None:
    """Translate a stormvogel Model and write it to a .umb file."""
    _write_umb(translate_to_umbi(model, ignore_unsupported_rewards), path)


def read_from_umb(
    path: str | Path,
    ignore_unsupported_rewards: bool = False,
    ignore_choice_annotations: bool = False,
    ignore_branch_annotations: bool = False,
) -> Model:
    """Read a .umb file and return it as a stormvogel Model."""
    return translate_to_stormvogel(
        _read_umb(path),
        ignore_unsupported_rewards,
        ignore_choice_annotations,
        ignore_branch_annotations,
    )
