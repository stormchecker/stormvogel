"""Tests for VisualizationBase, JSVisualization, MplVisualization, and JS validity.

The core test assembles the generated JS (NetworkWrapper class + init code) with
lightweight stubs for vis.js and the DOM, then executes the whole thing through
Node.js.  If any syntax error, unbalanced brace, or Python repr leaked into the
JS, Node will exit non-zero and the test fails with the actual JS engine error.
"""

import re
import shutil
import subprocess
import textwrap
from unittest.mock import patch

import matplotlib
import pytest

import stormvogel.examples as examples
import stormvogel.html_generation as html_gen
import stormvogel.layout as layout_mod
import stormvogel.model as model
from stormvogel.graph import node_key
from stormvogel.visualization import (
    JSVisualization,
    MplVisualization,
    VisualizationBase,
)
from stormvogel.show import show as show_fn, show_bird as show_bird_fn

NODE = shutil.which("node")

# ── Stubs ────────────────────────────────────────────────────────────
# Minimal stand-ins so that the generated JS can *execute*, not just parse.
# vis.DataSet, vis.Network, and document.getElementById all need to exist.

_VIS_STUB = textwrap.dedent("""\
    const _noop = () => ({});
    const _proxy = new Proxy({}, {
        get(_, prop) {
            // Any method call returns a callable that returns the proxy itself,
            // so chained calls like network.setData(...).on(...) don't crash.
            if (prop === Symbol.toPrimitive) return () => 0;
            return (..._args) => _proxy;
        }
    });

    const vis = {
        DataSet: function(items) {
            this._items = items;
            this.get = (id) => items.find(i => i.id === id) || {};
            this.update = _noop;
        },
        Network: function(_container, _data, _options) {
            return new Proxy({}, {
                get(_, prop) {
                    if (prop === 'getNodeAt') return () => undefined;
                    if (prop === 'getPosition') return () => ({x: 0, y: 0});
                    if (prop === 'getConnectedNodes') return () => [];
                    if (prop === 'getConnectedEdges') return () => [];
                    return (..._a) => _proxy;
                }
            });
        }
    };

    const document = {
        getElementById: (_id) => ({ clientWidth: 800, clientHeight: 600 })
    };

    // svgcanvas stubs (used by getSvg)
    function C2S() { return _proxy; }
    function Context() { return _proxy; }
""")


# ── Helpers ──────────────────────────────────────────────────────────


def _make_vis(m: model.Model) -> JSVisualization:
    """Create a JSVisualization without a server or display."""
    return JSVisualization(model=m, do_init_server=False)


def _run_js(code: str) -> subprocess.CompletedProcess:
    """Execute *code* through Node.js, returning the CompletedProcess."""
    assert NODE is not None, "Node.js not found"
    return subprocess.run(
        [NODE, "--eval", code],
        capture_output=True,
        text=True,
        timeout=10,
    )


def _assemble_full_js(vis: JSVisualization) -> str:
    """Build the complete JS program: stubs + NetworkWrapper class + init script.

    This mirrors what html_generation.generate_html() puts inside <script> tags,
    minus the 1.6 MB vis-network library (replaced by our stub).
    """
    network_wrapper_js = html_gen.generate_network_wrapper_js()
    init_js = html_gen.generate_init_js(
        vis._generate_node_js(),
        vis._generate_edge_js(),
        vis._get_options(),
        vis.name,
    )
    return f"{_VIS_STUB}\n{network_wrapper_js}\n{init_js}"


# ── Models under test ────────────────────────────────────────────────

ALL_MODELS = [
    examples.create_die_dtmc,
    examples.create_car_mdp,
    examples.create_lion_mdp,
    examples.create_monty_hall_mdp,
    examples.create_study_mdp,
    examples.create_nuclear_fusion_ctmc,
]

ALL_MODEL_IDS = [
    "die_dtmc",
    "car_mdp",
    "lion_mdp",
    "monty_hall_mdp",
    "study_mdp",
    "nuclear_fusion_ctmc",
]


# ── node_key unit tests ─────────────────────────────────────────────


def test_node_key_state():
    """node_key of a State is its UUID string — no spaces, no Python repr."""
    m = model.new_dtmc()
    key = node_key(m.initial_state)
    assert key == str(m.initial_state.state_id)
    assert " " not in key
    assert "<" not in key


def test_node_key_action_tuple():
    """node_key of (State, Action) base64url-encodes the action label."""
    from base64 import urlsafe_b64decode

    m = model.new_mdp()
    a = m.new_action("go")
    key = node_key((m.initial_state, a))
    uuid_part, encoded_label = key.split("__", 1)
    assert uuid_part == str(m.initial_state.state_id)
    assert urlsafe_b64decode(encoded_label).decode() == "go"
    # Key must be safe inside JS double-quoted strings
    assert '"' not in key
    assert "\\" not in key
    assert "\n" not in key


def test_node_key_special_chars():
    """Action labels with special chars produce safe, collision-free keys."""
    m = model.new_mdp()
    a1 = m.new_action('say "hello"')
    a2 = m.new_action("back\\slash")
    a3 = m.new_action("has__separator")

    k1 = node_key((m.initial_state, a1))
    k2 = node_key((m.initial_state, a2))
    k3 = node_key((m.initial_state, a3))

    # All keys must be JS-safe: no double-quotes, backslashes, or newlines
    for k in (k1, k2, k3):
        assert '"' not in k
        assert "\\" not in k
        assert "\n" not in k

    # All keys must be distinct
    assert len({k1, k2, k3}) == 3


# ── Core JS execution test ───────────────────────────────────────────


@pytest.mark.skipif(NODE is None, reason="Node.js not found on PATH")
@pytest.mark.parametrize("create_model", ALL_MODELS, ids=ALL_MODEL_IDS)
def test_generated_js_executes(create_model):
    """The generated JS parses and executes under Node.js without errors.

    This catches:
      - SyntaxError (unbalanced braces, missing quotes, illegal tokens)
      - ReferenceError from Python repr leaking in (e.g. `<generator …>`)
      - TypeError from malformed object literals
    """
    m = create_model()
    vis = _make_vis(m)
    js = _assemble_full_js(vis)
    result = _run_js(js)
    assert (
        result.returncode == 0
    ), f"Node.js failed (exit {result.returncode}):\n{result.stderr}"


# ── Node/edge ID referential integrity ───────────────────────────────


@pytest.mark.parametrize("create_model", ALL_MODELS, ids=ALL_MODEL_IDS)
def test_edge_ids_reference_existing_nodes(create_model):
    """Every from/to value in the edge JS must correspond to an id in the node JS."""
    m = create_model()
    vis = _make_vis(m)

    node_js = vis._generate_node_js()
    edge_js = vis._generate_edge_js()

    node_ids = set(re.findall(r'id:\s*"([^"]+)"', node_js))
    assert node_ids, "No node IDs found in generated JS"

    from_ids = set(re.findall(r'from:\s*"([^"]+)"', edge_js))
    to_ids = set(re.findall(r'to:\s*"([^"]+)"', edge_js))

    missing = (from_ids | to_ids) - node_ids
    assert not missing, f"Edges reference unknown node IDs: {missing}"


# ── Verify the DataSet contents via Node.js ──────────────────────────


@pytest.mark.skipif(NODE is None, reason="Node.js not found on PATH")
@pytest.mark.parametrize("create_model", ALL_MODELS, ids=ALL_MODEL_IDS)
def test_node_and_edge_counts_match_graph(create_model):
    """The number of nodes/edges emitted in JS matches the ModelGraph."""
    m = create_model()
    vis = _make_vis(m)
    js = _assemble_full_js(vis)

    # Append a probe that prints the DataSet sizes.
    probe = textwrap.dedent(f"""\
        const nw = nw_{vis.name};
        const nn = nw.nodes._items.length;
        const ne = nw.edges._items.length;
        process.stdout.write(nn + "," + ne);
    """)
    result = _run_js(js + "\n" + probe)
    assert (
        result.returncode == 0
    ), f"Node.js failed (exit {result.returncode}):\n{result.stderr}"

    js_nodes, js_edges = map(int, result.stdout.strip().split(","))
    assert js_nodes == len(
        vis.G.nodes
    ), f"JS has {js_nodes} nodes but ModelGraph has {len(vis.G.nodes)}"
    assert js_edges == len(
        vis.G.edges
    ), f"JS has {js_edges} edges but ModelGraph has {len(vis.G.edges)}"


# ── iframe variant ───────────────────────────────────────────────────


@pytest.mark.skipif(NODE is None, reason="Node.js not found on PATH")
def test_generate_iframe_valid_js():
    """generate_iframe wraps the same JS; verify the init script is still valid."""
    m = examples.create_car_mdp()
    vis = _make_vis(m)
    # The iframe embeds the full HTML.  We just need the init JS to be valid,
    # which is the same code path, so re-test via the direct route.
    js = _assemble_full_js(vis)
    result = _run_js(js)
    assert (
        result.returncode == 0
    ), f"Node.js failed (exit {result.returncode}):\n{result.stderr}"


# ── Regression: labels with special chars don't break JS ─────────────


@pytest.mark.skipif(NODE is None, reason="Node.js not found on PATH")
def test_special_characters_in_labels():
    """State labels containing backticks, quotes, or newlines must not break JS."""
    m = model.new_dtmc()
    init = m.initial_state
    init.add_label('has"quotes')
    init.add_label("has`backtick")
    init.set_choices([(1, m.new_state(labels=["has\nnewline"]))])
    m.add_self_loops()

    vis = _make_vis(m)
    js = _assemble_full_js(vis)
    result = _run_js(js)
    assert result.returncode == 0, (
        f"Node.js failed on special-char labels (exit {result.returncode}):\n"
        f"{result.stderr}"
    )


# ── VisualizationBase helpers ────────────────────────────────────────


def _dtmc_simple() -> model.Model:
    """A minimal 2-state DTMC for testing."""
    m = model.new_dtmc()
    s0 = m.initial_state
    s1 = m.new_state(labels=["end"])
    m.set_choices(s0, [(1, s1)])
    m.add_self_loops()
    return m


def _mdp_with_scheduler():
    """A simple MDP with a trivial scheduler for testing."""
    import stormvogel.result as result_mod

    m = model.new_mdp()
    s0 = m.initial_state
    a = m.new_action("go")
    s1 = m.new_state(labels=["end"])
    s0.set_choices([(a, s1)])
    m.add_self_loops()
    sched = result_mod.Scheduler(model=m, taken_actions={s0: a, s1: model.EmptyAction})
    return m, sched


# ── VisualizationBase: static helpers ───────────────────────────────


def test_blend_colors_identity():
    c = "#ff8040"
    assert VisualizationBase._blend_colors(c, c, 0.5) == c


def test_blend_colors_extremes():
    c1 = "#ffffff"
    c2 = "#000000"
    assert VisualizationBase._blend_colors(c1, c2, 1.0) == "#ffffff"
    assert VisualizationBase._blend_colors(c1, c2, 0.0) == "#000000"


def test_und_replaces_spaces():
    assert VisualizationBase._und("hello world") == "hello_world"


# ── VisualizationBase: format helpers ────────────────────────────────


def test_format_number_hidden():
    m = _dtmc_simple()
    lay = layout_mod.Layout()
    lay.layout["numbers"]["visible"] = False
    vis = VisualizationBase(model=m, layout=lay)
    assert vis._format_number(0.5) == ""


def test_format_result_none_when_no_result():
    m = _dtmc_simple()
    vis = VisualizationBase(model=m)
    assert vis._format_result(m.initial_state) == ""


def test_format_result_with_result():
    import stormvogel.result as result_mod

    m = _dtmc_simple()
    res = result_mod.Result(
        model=m,
        values={s: 0.5 for s in m.states},
    )
    vis = VisualizationBase(model=m, result=res)
    s = m.initial_state
    formatted = vis._format_result(s)
    assert formatted != ""  # some result is shown


def test_format_observations_dtmc_returns_empty():
    """DTMC doesn't support observations — should return ''."""
    m = _dtmc_simple()
    vis = VisualizationBase(model=m)
    assert vis._format_observations(m.initial_state) == ""


def test_format_observations_pomdp():
    pomdp = model.new_pomdp()
    obs = pomdp.new_observation("o1")
    s = pomdp.new_state(observation=obs)
    pomdp.add_self_loops()
    vis = VisualizationBase(model=pomdp)
    result = vis._format_observations(s)
    assert "o1" in result


def test_group_state_default():
    m = _dtmc_simple()
    vis = VisualizationBase(model=m)
    group = vis._group_state(m.initial_state, "states")
    assert group == "states"


def test_group_state_with_label():
    m = model.new_dtmc()
    s0 = m.initial_state
    s0.add_label("init")
    m.add_self_loops()
    lay = layout_mod.Layout()
    lay.add_active_group("init")
    vis = VisualizationBase(model=m, layout=lay)
    assert vis._group_state(s0, "states") == "init"


def test_group_action_no_scheduler():
    m, _ = _mdp_with_scheduler()
    vis = VisualizationBase(model=m)
    a = model.EmptyAction
    group = vis._group_action(m.initial_state, a, "actions")
    assert group == "actions"


def test_group_action_with_scheduler():
    m, sched = _mdp_with_scheduler()
    vis = VisualizationBase(model=m, scheduler=sched)
    s0 = m.initial_state
    scheduled_action = sched.get_action_at_state(s0)
    group = vis._group_action(s0, scheduled_action, "actions")
    assert group == "scheduled_actions"


def test_format_rewards_non_empty_action():
    m = _dtmc_simple()
    vis = VisualizationBase(model=m)
    # Any action other than EmptyAction should return ""
    a = m.new_action("x")
    assert vis._format_rewards(m.initial_state, a) == ""


def test_format_rewards_empty_action_with_reward():
    m = _dtmc_simple()
    rm = m.new_reward_model("r1")
    rm.set_state_reward(m.initial_state, 5.0)
    vis = VisualizationBase(model=m)
    result = vis._format_rewards(m.initial_state, model.EmptyAction)
    assert "5" in result or "r1" in result


def test_create_state_properties_structure():
    m = _dtmc_simple()
    vis = VisualizationBase(model=m)
    props = vis._create_state_properties(m.initial_state)
    assert "label" in props and "group" in props and "color" in props


def test_create_state_properties_with_color_blending():
    import stormvogel.result as result_mod

    m = _dtmc_simple()
    states = list(m.states)
    res = result_mod.Result(
        model=m, values={s: float(i + 1) for i, s in enumerate(states)}
    )
    lay = layout_mod.Layout()
    lay.layout["results"]["result_colors"] = True
    vis = VisualizationBase(model=m, layout=lay, result=res)
    # states[-1] has the highest value (== len(states)), so factor == 1.0
    # and the blended color must equal max_result_color exactly.
    props = vis._create_state_properties(states[-1])
    color = props["color"]
    assert isinstance(color, str), "color must be a string"
    assert color.startswith("#") and len(color) == 7, f"expected #rrggbb, got {color!r}"
    assert color == lay.layout["results"]["max_result_color"]


def test_create_action_properties():
    m, _ = _mdp_with_scheduler()
    vis = VisualizationBase(model=m)
    s0 = m.initial_state
    action = list(s0.available_actions())[0]
    props = vis._create_action_properties(s0, action)
    assert "label" in props and "model_action" in props


def test_create_transition_properties_found():
    m = _dtmc_simple()
    vis = VisualizationBase(model=m)
    s0 = m.initial_state
    states = list(m.states)
    s1 = states[1] if states[0] == s0 else states[0]
    props = vis._create_transition_properties(s0, model.EmptyAction, s1)
    assert "label" in props


def test_create_transition_properties_no_transition_to_target():
    m = _dtmc_simple()
    vis = VisualizationBase(model=m)
    s0 = m.initial_state
    # s0 -> s0 self-loop via add_self_loops, but create a fresh unrelated state
    s_new = m.new_state(labels=["x"])
    # Don't add transitions from s0 to s_new
    props = vis._create_transition_properties(s0, model.EmptyAction, s_new)
    # s_new is not a target, so we get empty dict back
    assert props == {}


def test_create_transition_properties_no_transitions():
    m = model.new_dtmc(create_initial_state=False)
    s = m.new_state()
    # No transitions for this state — get_outgoing_transitions returns None
    vis = VisualizationBase(model=m)
    props = vis._create_transition_properties(s, model.EmptyAction, s)
    assert props == {}


# ── JSVisualization extended ─────────────────────────────────────────


def test_jsvis_explore_mode():
    m = _dtmc_simple()
    vis = _make_vis(m)
    vis.enable_exploration_mode(m.initial_state)
    assert vis.layout.layout["misc"]["explore"] is True
    # Node JS should contain hidden: true for non-initial nodes
    node_js = vis._generate_node_js()
    assert "hidden: true" in node_js


def test_jsvis_generate_iframe_contains_iframe_tag():
    m = _dtmc_simple()
    vis = _make_vis(m)
    iframe_html = vis.generate_iframe()
    assert "<iframe" in iframe_html


def test_jsvis_set_options():
    import json

    m = _dtmc_simple()
    vis = _make_vis(m)
    new_opts = json.dumps({"misc": {"enable_physics": False}})
    vis.set_options(new_opts)
    assert vis.layout.layout["misc"]["enable_physics"] is False


def test_jsvis_export_html(tmp_path):
    m = _dtmc_simple()
    vis = _make_vis(m)
    out = str(tmp_path / "out")
    vis.export("html", filename=out)
    assert (tmp_path / "out.html").exists()


def test_jsvis_export_iframe(tmp_path):
    m = _dtmc_simple()
    vis = _make_vis(m)
    out = str(tmp_path / "out")
    vis.export("iframe", filename=out)
    assert (tmp_path / "out.html").exists()


def test_jsvis_export_unknown_raises():
    m = _dtmc_simple()
    vis = _make_vis(m)
    with pytest.raises(RuntimeError):
        vis.export("jpeg", filename="out")


def test_jsvis_highlight_state():
    m = _dtmc_simple()
    vis = _make_vis(m)
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.highlight_state(m.initial_state, color="blue")


def test_jsvis_highlight_state_not_in_graph_raises():
    m1 = _dtmc_simple()
    m2 = model.new_dtmc()
    vis = _make_vis(m1)
    with pytest.raises(AssertionError):
        vis.highlight_state(m2.initial_state, color="blue")


def test_jsvis_highlight_action():
    m, _ = _mdp_with_scheduler()
    vis = _make_vis(m)
    s0 = m.initial_state
    action = list(s0.available_actions())[0]
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.highlight_action(s0, action, color="green")


def test_jsvis_highlight_action_unknown_no_crash():
    """highlight_action with an unknown action silently does nothing."""
    m1, _ = _mdp_with_scheduler()
    m2 = model.new_mdp()
    vis = _make_vis(m1)
    a_foreign = m2.new_action("foreign")
    # Should not raise, just silently ignores unknown action
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.highlight_action(m1.initial_state, a_foreign, color="red")


def test_jsvis_highlight_state_set():
    m = _dtmc_simple()
    vis = _make_vis(m)
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.highlight_state_set(set(m.states), color="yellow")


def test_jsvis_highlight_decomposition_with_colors():
    m = _dtmc_simple()
    vis = _make_vis(m)
    states = set(m.states)
    decomp = [(states, set())]
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.highlight_decomposition(decomp, colors=["#aabbcc"])


def test_jsvis_highlight_decomposition_random_colors():
    m = _dtmc_simple()
    vis = _make_vis(m)
    states = set(m.states)
    decomp = [(states, set())]
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.highlight_decomposition(decomp, colors=None)


def test_jsvis_clear_highlighting():
    m = _dtmc_simple()
    vis = _make_vis(m)
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.clear_highlighting()


def test_jsvis_show():
    m = _dtmc_simple()
    vis = _make_vis(m)
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.show()


def test_jsvis_show_iframe():
    m = _dtmc_simple()
    vis = JSVisualization(model=m, use_iframe=True, do_init_server=False)
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.show()


def test_jsvis_update():
    m = _dtmc_simple()
    vis = _make_vis(m)
    with patch("IPython.display.display"), patch("IPython.display.clear_output"):
        vis.update()


# ── MplVisualization ─────────────────────────────────────────────────

matplotlib.use("Agg")


def test_mpl_vis_creates():
    m = _dtmc_simple()
    vis = MplVisualization(model=m)
    assert vis.model is m
    # The underlying ModelGraph must have one node per model state.
    assert len(vis.G.nodes) == len(m.states)


def test_mpl_vis_with_title():
    m = _dtmc_simple()
    vis = MplVisualization(model=m, title="My Title")
    assert vis.title == "My Title"


def test_mpl_vis_highlight_state():
    m = _dtmc_simple()
    vis = MplVisualization(model=m)
    vis.highlight_state(m.initial_state, color="red")
    assert m.initial_state in vis._highlights


def test_mpl_vis_highlight_state_not_in_graph_raises():
    m1 = _dtmc_simple()
    m2 = model.new_dtmc()
    vis = MplVisualization(model=m1)
    with pytest.raises(AssertionError):
        vis.highlight_state(m2.initial_state, color="red")


def test_mpl_vis_highlight_action():
    m, _ = _mdp_with_scheduler()
    vis = MplVisualization(model=m)
    s0 = m.initial_state
    action = list(s0.available_actions())[0]
    vis.highlight_action(s0, action, color="blue")
    assert (s0, action) in vis._highlights


def test_mpl_vis_highlight_edge():
    m = _dtmc_simple()
    vis = MplVisualization(model=m)
    states = list(m.states)
    vis.highlight_edge(states[0], states[1], color="green")
    assert (states[0], states[1]) in vis._edge_highlights


def test_mpl_vis_clear_highlighting():
    m = _dtmc_simple()
    vis = MplVisualization(model=m)
    vis.highlight_state(m.initial_state, color="red")
    vis.clear_highlighting()
    assert len(vis._highlights) == 0
    assert len(vis._edge_highlights) == 0


def test_mpl_vis_highlight_scheduler():
    m, sched = _mdp_with_scheduler()
    vis = MplVisualization(model=m, scheduler=sched)
    # Scheduler highlight is applied in __init__ when scheduler is set
    assert len(vis._highlights) > 0


def test_mpl_vis_update_twice():
    """Calling update() twice reuses the same figure."""
    import matplotlib.pyplot as plt

    m = _dtmc_simple()
    vis = MplVisualization(model=m)
    fig1 = vis.update()
    fig2 = vis.update()
    assert fig1 is fig2
    plt.close("all")


# ── show.py ──────────────────────────────────────────────────────────


def test_show_js_engine(tmp_path, monkeypatch):
    """show() with engine='js' returns a JSVisualization."""
    monkeypatch.chdir(tmp_path)  # so model.html is written to a temp dir
    m = _dtmc_simple()
    with (
        patch("IPython.display.display"),
        patch("IPython.display.clear_output"),
        patch("IPython.display.HTML"),
    ):
        vis = show_fn(m, engine="js", do_init_server=False)
    assert isinstance(vis, JSVisualization)


def test_show_mpl_engine():
    """show() with engine='mpl' returns a MplVisualization."""
    import matplotlib.pyplot as plt

    m = _dtmc_simple()
    with patch("matplotlib.pyplot.show"):
        vis = show_fn(m, engine="mpl", do_init_server=False)
    assert isinstance(vis, MplVisualization)
    plt.close("all")


def test_show_unknown_engine_returns_none():
    """show() with an unknown engine prints an error and returns None."""
    m = _dtmc_simple()
    result = show_fn(m, engine="unknown", do_init_server=False)
    assert result is None


def test_show_with_pos_function(tmp_path, monkeypatch):
    """show() with pos_function applies networkx positions."""
    import networkx as nx

    monkeypatch.chdir(tmp_path)
    m = _dtmc_simple()
    with (
        patch("IPython.display.display"),
        patch("IPython.display.clear_output"),
        patch("IPython.display.HTML"),
    ):
        vis = show_fn(
            m,
            engine="js",
            do_init_server=False,
            pos_function=nx.random_layout,
        )
    assert vis is not None


def test_show_with_iframe(tmp_path, monkeypatch):
    """show() with use_iframe=True uses iframe HTML."""
    monkeypatch.chdir(tmp_path)
    m = _dtmc_simple()
    with (
        patch("IPython.display.display"),
        patch("IPython.display.clear_output"),
        patch("IPython.display.HTML"),
    ):
        vis = show_fn(m, engine="js", do_init_server=False, use_iframe=True)
    assert isinstance(vis, JSVisualization)


def test_show_bird(tmp_path, monkeypatch):
    """show_bird() returns a JSVisualization of a bird model."""
    monkeypatch.chdir(tmp_path)
    with (
        patch("IPython.display.display"),
        patch("IPython.display.clear_output"),
        patch("IPython.display.HTML"),
    ):
        vis = show_bird_fn()
    assert isinstance(vis, JSVisualization)
