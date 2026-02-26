from stormvogel.layout import Layout
import numpy as np
import os
import json

from stormvogel.rdict import merge_dict


def test_layout_loading():
    """Tests if str(Layout) returns the correctly loaded json string."""
    with open(os.path.join(os.getcwd(), "stormvogel/layouts/default.json")) as f:
        default_str = f.read()
    with open(os.path.join(os.getcwd(), "tests/test_layout.json")) as f:
        test_str = f.read()
    default_dict = json.loads(default_str)
    test_dict = json.loads(test_str)
    expected = json.dumps(
        merge_dict(default_dict, test_dict), indent=2
    )  # We can use Layout.merge_dict since we have already tested it.
    actual = str(Layout("tests/test_layout.json"))

    assert expected == actual


def test_empty_layout_loading():
    """Same as previous test but now with an empty layout."""
    with open(os.path.join(os.getcwd(), "stormvogel/layouts/default.json")) as f:
        default_str = f.read()
    default_dict = json.loads(default_str)

    expected = json.dumps(default_dict, indent=2)
    actual = str(Layout())
    assert expected == actual


def test_layout_saving():
    """Tests if the saved layout from Layout.save() is equal to str(Layout)."""
    layout = Layout("tests/test_layout.json")
    try:
        os.remove(
            os.path.join(os.getcwd(), "tests/saved_test_layout.json")
        )  # First remove the existing file.
    except FileNotFoundError:
        pass  # The file did not exist yet.
    layout.save("tests/saved_test_layout.json")
    with open(os.path.join(os.getcwd(), "tests/saved_test_layout.json")) as f:
        saved_string = f.read()
    assert saved_string == str(layout)


def test_nx_pos():
    pos = {0: np.array([0, 0]), 1: np.array([1, 1])}
    layout = Layout("tests/test_layout.json").set_nx_pos(pos, scale=1)
    assert layout.layout["positions"] == {
        "0": {"x": 0.0, "y": 0.0},
        "1": {"x": 1.0, "y": 1.0},
    }
