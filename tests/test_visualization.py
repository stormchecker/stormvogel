"""Tests that the JS visualization generates syntactically valid, executable JavaScript.

The core test assembles the generated JS (NetworkWrapper class + init code) with
lightweight stubs for vis.js and the DOM, then executes the whole thing through
Node.js.  If any syntax error, unbalanced brace, or Python repr leaked into the
JS, Node will exit non-zero and the test fails with the actual JS engine error.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import stormvogel.examples as examples
import stormvogel.html_generation as html_gen
import stormvogel.model as model
from stormvogel.graph import node_key
from stormvogel.visualization import JSVisualization

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
    """node_key of (State, Action) encodes the state UUID and action label."""
    m = model.new_mdp()
    a = m.new_action("go")
    key = node_key((m.initial_state, a))
    assert str(m.initial_state.state_id) in key
    assert "go" in key
    assert " " not in key


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
