"""Tests for stormvogel.html_generation."""

from stormvogel.html_generation import generate_init_js, generate_network_wrapper_js


def test_generate_init_js_embeds_arguments():
    js = generate_init_js("node1", "edge1", '{"x": 1}', "mynet")
    # nodes_js and edges_js must be embedded inside the DataSet constructors
    assert "new vis.DataSet([node1])" in js
    assert "new vis.DataSet([edge1])" in js
    # options_js must be assigned verbatim
    assert 'var options_local = {"x": 1}' in js
    # container must use the given name
    assert 'document.getElementById("mynet")' in js
    # NetworkWrapper instantiation must pass all four locals
    assert (
        "new NetworkWrapper(nodes_local, edges_local, options_local, container_local)"
        in js
    )
    # result stored under nw_{name}
    assert "var nw_mynet =" in js


def test_generate_init_js_name_scopes_variable():
    js_a = generate_init_js("[]", "[]", "{}", "alpha")
    js_b = generate_init_js("[]", "[]", "{}", "beta")
    assert "nw_alpha" in js_a and "nw_beta" not in js_a
    assert "nw_beta" in js_b and "nw_alpha" not in js_b


def test_generate_network_wrapper_js_constructor_wires_up_fields():
    js = generate_network_wrapper_js()
    assert "this.nodes = nodes" in js
    assert "this.edges = edges" in js
    assert "this.network = new vis.Network(" in js


def test_generate_network_wrapper_js_methods_present():
    js = generate_network_wrapper_js()
    # Method signatures, not just names
    assert "makeNeighborsVisible(homeId)" in js
    assert "setNodeColor(id, color)" in js
    assert "getSvg()" in js
