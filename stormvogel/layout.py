"""Store and manage loading/saving of the layout file and the schema file.

The schema file may be used by the LayoutEditor to create an editor for this Layout.
The Layout object also provides methods to manipulate the layout and the schema,
such as setting node positions from a NetworkX layout.
"""

from typing import Any, Self

import stormvogel.rdict

import os
import json
import copy

PACKAGE_ROOT_DIR = os.path.dirname(
    os.path.realpath(__file__)
)  # Get the stormvogel/ directory.


class Layout:
    """Store the layout and schema for a visualization.

    Upon creation, the layout and schema dicts are loaded from
    ``layouts/default.json`` and ``layouts/schema.json``, unless specified otherwise.
    Load a custom layout file by setting either *path* or *path_relative*, or provide
    a custom layout dict instead.

    :param path: Path to a custom layout file. Leave as ``None`` for the default layout.
    :param path_relative: If ``True``, stormvogel looks for a custom layout file
        relative to the current working directory.
    :param layout_dict: If set, this dictionary is used as the layout instead of the
        file specified in *path*. Missing keys are filled from ``layouts/default.json``.
    """

    def __init__(
        self,
        path: str | None = None,
        path_relative: bool = True,
        layout_dict: dict | None = None,
    ):
        # Open the default layout file.
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

    def load_schema(self) -> None:
        """Load in the schema. Used for the layout editor. Stored as self.schema."""
        with open(os.path.join(PACKAGE_ROOT_DIR, "layouts/schema.json")) as f:
            schema_str = f.read()
        self.schema = json.loads(schema_str)

    def load(self, path: str | None = None, path_relative: bool = True) -> None:
        """Load the layout and schema file at the specified path.

        They are stored as :attr:`layout` and :attr:`schema` respectively.

        :param path: Path to the layout file, or ``None`` for the default layout.
        :param path_relative: If ``True``, *path* is resolved relative to the current
            working directory.
        """
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
        """Make a group active if it is not already.

        The user can specify which groups of states can be edited separately.
        Such groups are referred to as active groups.

        :param group: Name of the group to activate.
        """
        if group not in self.layout["edit_groups"]["groups"]:
            self.layout["edit_groups"]["groups"].append(group)

    def remove_active_group(self, group: str) -> None:
        """Make a group inactive if it is not already.

        :param group: Name of the group to deactivate.
        """
        if group in self.layout["edit_groups"]["groups"]:
            self.layout["edit_groups"]["groups"].remove(group)

    def set_possible_groups(self, groups: set[str]) -> None:
        """Set the groups of states that the user can choose to make active.

        These appear under ``edit_groups`` in the layout editor.

        :param groups: Set of group names to make available.
        """
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
        """Save this layout as a JSON file.

        :param path: Path to the layout file. Must end in ``.json``.
        :param path_relative: If ``True``, *path* is resolved relative to the current
            working directory.
        :raises RuntimeError: If the filename does not end in ``.json``.
        :raises OSError: If the file cannot be written.
        """
        if path[-5:] != ".json":
            raise RuntimeError("File name should end in .json")
        if path_relative:
            complete_path = os.path.join(os.getcwd(), path)
        else:
            complete_path = path
        with open(complete_path, "w") as f:
            json.dump(self.layout, f, indent=2)

    def set_value(self, path: list[str], value: Any) -> None:
        """Set a value in the layout.

        Also works if a key in the path does not exist.

        :param path: List of keys forming the path to the value.
        :param value: The value to set.
        """
        stormvogel.rdict.rset(self.layout, path, value, create_new_keys=True)

    def __str__(self) -> str:
        return json.dumps(self.layout, indent=2)

    def copy_settings(self) -> None:
        """Copy some settings from one place in the layout to another place in the layout.
        They differ because visjs requires for them to be arranged a certain way which is not nice for an editor.
        """
        self.layout["physics"] = self.layout["misc"]["enable_physics"]

    def set_nx_pos(self, pos: dict, scale: float = 500) -> Self:
        """Apply NetworkX layout positions to this layout and disable physics.

        :param pos: Dictionary of node positions from a NetworkX graph.
        :param scale: Scaling factor for the positions.
        :returns: This :class:`Layout` instance, for chaining.
        """
        from stormvogel.graph import node_key

        self.set_value(["misc", "enable_physics"], False)
        self.set_value(["physics"], False)
        self.set_value(
            ["positions"],
            {
                node_key(k): {"x": float(x * scale), "y": float(y * scale)}
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
    default.layout["misc"]["explore"] = True
    return default


def SV():
    return Layout(
        os.path.join(PACKAGE_ROOT_DIR, "layouts/sv.json"), path_relative=False
    )


def LTS():
    default = DEFAULT()
    default.layout["numbers"]["visible"] = False
    return default
