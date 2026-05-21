from stormvogel.layout import Layout, EXPLORE, SV, LTS
import numpy as np
import os
import json
import pytest

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
    import stormvogel.model as model
    from stormvogel.graph import ModelGraph, node_key

    m = model.new_dtmc()
    s0 = m.initial_state
    s1 = m.new_state(labels=["end"])
    m.set_choices(s0, [(1, s1)])
    m.add_self_loops()

    G = ModelGraph.from_model(m)
    nodes = list(G.nodes)
    pos = {nodes[0]: np.array([0, 0]), nodes[1]: np.array([1, 1])}
    layout = Layout("tests/test_layout.json").set_nx_pos(pos, scale=1)
    assert layout.layout["positions"] == {
        node_key(nodes[0]): {"x": 0.0, "y": 0.0},
        node_key(nodes[1]): {"x": 1.0, "y": 1.0},
    }


def test_layout_from_dict():
    layout = Layout(layout_dict={"misc": {"enable_physics": False}})
    assert layout.layout["misc"]["enable_physics"] is False


def test_add_and_remove_active_group():
    layout = Layout()
    layout.add_active_group("group_a")
    assert "group_a" in layout.layout["edit_groups"]["groups"]
    # Adding again should be idempotent
    layout.add_active_group("group_a")
    assert layout.layout["edit_groups"]["groups"].count("group_a") == 1

    layout.remove_active_group("group_a")
    assert "group_a" not in layout.layout["edit_groups"]["groups"]
    # Removing non-existent should not raise
    layout.remove_active_group("group_a")


def test_set_value_creates_new_keys():
    layout = Layout()
    layout.set_value(["custom", "nested", "key"], 42)
    assert layout.layout["custom"]["nested"]["key"] == 42


def test_copy_settings_propagates_physics():
    layout = Layout()
    layout.layout["misc"]["enable_physics"] = False
    layout.copy_settings()
    assert layout.layout["physics"] is False


def test_save_raises_for_wrong_extension(tmp_path):
    layout = Layout()
    with pytest.raises(RuntimeError, match=".json"):
        layout.save(str(tmp_path / "bad_name.txt"), path_relative=False)


def test_explore_layout():
    layout = EXPLORE()
    assert layout.layout["misc"]["explore"] is True


def test_sv_layout():
    layout = SV()
    assert layout is not None
    assert "misc" in layout.layout


def test_lts_layout():
    layout = LTS()
    assert layout.layout["numbers"]["visible"] is False
