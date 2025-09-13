"""Javascript code generation functions, used in network.py. Depends on having vis-network.min.js in the stormvogel foder."""

from stormvogel.layout import PACKAGE_ROOT_DIR


def generate_init_js(nodes_js: str, edges_js: str, options_js: str, name: str) -> str:
    return f"""//js
    var nodes_local = new vis.DataSet([{nodes_js}]);
    var edges_local = new vis.DataSet([{edges_js}]);
    var options_local = {options_js};
    var container_local = document.getElementById("{name}");
    var nw_{name} = new NetworkWrapper(nodes_local, edges_local, options_local, container_local)
    """


# An html template on which a Network is based.
def generate_html(
    nodes_js: str, edges_js: str, options_js: str, name: str, width: int, height: int
):
    """Generate HTML that renders the network.
    You should be able to locate the NetworkWrapper object as nw_{name},
    and nw_{name} has a field that is the visjs network itself."""
    with open(PACKAGE_ROOT_DIR + "/vis-network-9.1.9-patched.js") as f:
        visjs_library = f.read()
    with open(PACKAGE_ROOT_DIR + "/svgcanvas.js") as f:
        svg_canvas_library = f.read()
    # Note that double brackets {{ }} are used to escape characters '{' and '}'
    return f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Network</title>
    <script>{visjs_library}</script>
    <script>{svg_canvas_library}</script>
    <style type="text/css">
      #{name} {{
        width: {width}px;
        height: {height}px;
        border: 1px solid lightgray;
      }}
    </style>
  </head>
  <body>
    <div id="{name}"></div>
    <script type="text/javascript">
        {generate_network_wrapper_js()}
    </script>
    <script type="text/javascript">
        {generate_init_js(nodes_js, edges_js, options_js, name)}
    </script>
  </body>
</html>
"""


def generate_network_wrapper_js():
    # Javascript code for finding the container and initializing the network
    # Having a separate NewtorkWrapper object allows us to have multiple networks in one notebook without them interfering.
    return """//js
class NetworkWrapper {
  constructor(nodes, edges, options, container) {
    this.nodes = nodes;
    this.edges = edges;
    this.options = options;
    this.container = container;
    this.data = {
      nodes: nodes,
      edges: edges,
    };
    this.network = new vis.Network(container, this.data, options);
    var this_ = this; // Events will not work if you use 'this' directly :))))) No idea why.
    this_.network.setData(this_.data);

    // Set user-triggered events.
    this.network.on( 'click', function(properties) {
      var nodeId = this_.network.getNodeAt({x:properties.event.srcEvent.offsetX, y:properties.event.srcEvent.offsetY});
      this_.makeNeighborsVisible(nodeId);
    });
    this.network.on( 'doubleClick', function(properties) {
        this_.network.setData(this_.data);
    });
  }

  setNodeColor(id, color) {
    var node = this.nodes.get(id);
    node["x"] = this.network.getPosition(id)["x"];
    node["y"] = this.network.getPosition(id)["y"];
    node["color"] = color;
    this.nodes.update(node);
  }

  makeNeighborsVisible(homeId) {
    if (homeId === undefined) {
      return;
    }
    var homeNode = this.nodes.get(homeId);

    // Make outgoing nodes visible
    var nodeIds = this.network.getConnectedNodes(homeId, "to");
    for (let i = 0; i < nodeIds.length; i++) {
      var toNodeId = nodeIds[i];
      var toNode = this.nodes.get(toNodeId);
      if (toNode["hidden"]) {
        toNode["hidden"] = false;
        toNode["physics"] = true;
        toNode["x"] = this.network.getPosition(homeId)["x"];
        toNode["y"] = this.network.getPosition(homeId)["y"];
        this.nodes.update(toNode);
      }
    }
    // Make edges visible, if both of the nodes are also visible
    var edgeIds = this.network.getConnectedEdges(homeId);
    for (let i = 0; i < edgeIds.length; i++) {
      var edgeId = edgeIds[i];
      var edge = this.edges.get(edgeId);
      var fromNode = this.nodes.get(edge.from);
      var toNode = this.nodes.get(edge.to);
      if ((! fromNode["hidden"]) && (! toNode["hidden"])) {
        edge["hidden"] = false;
        edge["physics"] = true;
        this.edges.update(edge);
      }
    }
  }

  getSvg() {
    // Export the network as an svg, return the raw svg file.
    // Most of this code is from Justin Harrell's vis-svg https://github.com/justinharrell/vis-svg
    let network = this.network;
    var networkContainer = network.body.container;
    try { // For whatever strange reason, if you enable iframe, it only works with Context and it only works with C2S otherwise...
      var ctx = new C2S({width: networkContainer.clientWidth, height: networkContainer.clientHeight, embedImages: true});
    } catch(e) {
      var ctx = new Context({width: networkContainer.clientWidth, height: networkContainer.clientHeight, embedImages: true});
    }

    var canvasProto = network.canvas.__proto__;
    var currentGetContext = canvasProto.getContext;
    canvasProto.getContext = function()
    {
        return ctx;
    }
    var svgOptions = {
        nodes: {
            shapeProperties: {
                interpolation: false //so images are not scaled svg will get full image
            },
            scaling: { label: { drawThreshold : 0} },
            font:{color:'#000000'}
        },
        edges: {
            scaling: { label: { drawThreshold : 0} }
        }
    };

    network.setOptions(svgOptions);
    network.redraw();
    canvasProto.getContext = currentGetContext;
    var svg = ctx.getSerializedSvg();
    return svg;
  }
};
"""
