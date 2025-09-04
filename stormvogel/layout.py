"""Contains the code responsible for saving/loading layouts and modifying them interactively."""

from typing import Any, Self

import numpy.typing as npt
import stormvogel.rdict

import os
import json
import copy

PACKAGE_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


class Layout:
    """Responsible for loading/saving layout jsons."""

    def __init__(
        self,
        path: str | None = None,
        path_relative: bool = True,
        layout_dict: dict | None = None,
    ) -> None:
        """Load a new Layout from a json file.
        Whenever keys are not present in the provided json file, their default values are used instead
        as specified in DEFAULTS (=layouts/default.json).

        Args:
            path (str, optional): Path to your custom layout file.
                Leave to None for an empty/default layout. Defaults to None.
            path_relative (bool, optional): If set to True, then stormvogel will look for a custom layout
                file relative to the current working directory. Defaults to True.
            layout_dict (dict, optional): If set, this dictionary is used as the layout instead of the
                file specified in path.
        """
        with open(os.path.join(PACKAGE_ROOT_DIR, "layouts/default.json")) as f:
            default_str = f.read()
        self.default_dict: dict = json.loads(default_str)

        if layout_dict is None:
            self.load(path, path_relative)
        else:
            self.layout: dict = stormvogel.rdict.merge_dict(
                self.default_dict, layout_dict
            )
            self.load_schema()

    def load_schema(self):
        """Load in the schema. Used for the layout editor. Stored as self.schema."""
        with open(os.path.join(PACKAGE_ROOT_DIR, "layouts/schema.json")) as f:
            schema_str = f.read()
        self.schema = json.loads(schema_str)

    def load(self, path: str | None = None, path_relative: bool = True):
        """Load the layout and schema file at the specified path.
        They are stored as self.layout and self.schema respectively."""
        if path is None:
            self.layout: dict = self.default_dict
        else:
            if path_relative:
                complete_path = os.path.join(os.getcwd(), path)
            else:
                complete_path = path
            with open(complete_path) as f:
                parsed_str = f.read()
            parsed_dict = json.loads(parsed_str)
            # Combine the parsed dict with default to fill missing keys as default values.
            self.layout: dict = stormvogel.rdict.merge_dict(
                self.default_dict, parsed_dict
            )
        self.load_schema()

    def add_active_group(self, group: str) -> None:
        """Make a group active if it is not already."""
        if group not in self.layout["edit_groups"]["groups"]:
            self.layout["edit_groups"]["groups"].append(group)

    def remove_active_group(self, group: str) -> None:
        """Make a group inactive if it is not already."""
        if group in self.layout["edit_groups"]["groups"]:
            self.layout["edit_groups"]["groups"].remove(group)

    def set_possible_groups(self, groups: set[str]):
        """Set the groups of states that the user can choose from under edit_groups."""
        self.schema["edit_groups"]["groups"]["__kwargs"]["allowed_tags"] = list(groups)

        # Save changes to the schema. The visualization object will handle putting nodes into the correct groups.
        groups2 = self.layout["edit_groups"]["groups"]
        self.schema[
            "groups"
        ] = {}  # empty the schema groups, to clear existing groups that we may not want
        for g in groups2:
            # For the settings themselves, we need to manually copy everything.
            layout_group_macro = copy.deepcopy(
                self.layout["__fake_macros"]["__group_macro"]
            )
            # Merge the macro with any existing changes.
            existing = self.layout["groups"][g] if g in self.layout["groups"] else {}
            self.layout["groups"][g] = stormvogel.rdict.merge_dict(
                layout_group_macro, existing
            )

            # For the schema, dict_editor already handles macros, so there is no need to do it manually here.
            if g not in self.schema["groups"]:
                self.schema["groups"][g] = {"__use_macro": "__group_macro"}

    def save(self, path: str, path_relative: bool = True) -> None:
        """Save this layout as a json file. Raises runtime error if a filename does not end in json, and OSError if file not found.

        Args:
            path (str): Path to your layout file.
            path_relative (bool, optional): If set to true, then stormvogel will create a custom layout
                file relative to the current working directory. Defaults to True.
        """
        if path[-5:] != ".json":
            raise RuntimeError("File name should end in .json")
        if path_relative:
            complete_path = os.path.join(os.getcwd(), path)
        else:
            complete_path = path
        with open(complete_path, "w") as f:
            json.dump(self.layout, f, indent=2)

    def set_value(self, path: list[str], value: Any):
        """Set a value in the layout. Also works if a key in the path does not exist."""
        stormvogel.rdict.rset(self.layout, path, value, create_new_keys=True)

    def __str__(self) -> str:
        return json.dumps(self.layout, indent=2)

    def copy_settings(self):
        """Copy some settings from one place in the layout to another place in the layout.
        They differ because visjs requires for them to be arranged a certain way which is not nice for an editor."""
        self.layout["physics"] = self.layout["misc"]["enable_physics"]

    def set_nx_pos(self, pos: dict[int, npt.NDArray], scale: float = 500) -> Self:
        """Apply NetworkX layout positions to this layout and set physics to false.

        Args:
            pos (dict[int, NDArray]): A dictionary of node positions from a NetworkX graph.
            scale (float): Scaling factor for the positions. Defaults to 500.
        """
        self.set_value(["misc", "enable_physics"], False)
        self.set_value(["physics"], False)
        self.set_value(
            ["positions"],
            {
                k: {"x": float(x * scale), "y": float(y * scale)}
                for k, (x, y) in pos.items()
            },
        )
        return self


# Define template layouts.
def DEFAULT():
    return Layout(
        os.path.join(PACKAGE_ROOT_DIR, "layouts/default.json"), path_relative=False
    )


def EXPLORE():
    default = DEFAULT()
    default.layout.layout["misc"]["explore"] = True
    return default


def SV():
    return Layout(
        os.path.join(PACKAGE_ROOT_DIR, "layouts/sv.json"), path_relative=False
    )
