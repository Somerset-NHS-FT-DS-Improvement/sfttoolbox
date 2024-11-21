__all__ = ["generate_sankey"]

import networkx as nx
import plotly.graph_objects as go


def generate_sankey(G: nx.Graph) -> None:
    """
    Generate and display a Sankey diagram from a NetworkX graph.

    This function takes a NetworkX graph `G` and creates an interactive Sankey diagram using Plotly.
    The graph `G` should have nodes and edges with specific attributes as follows:

    Node attributes:
    - "color": The color of the node.

    Edge attributes:
    - "value": The value (weight) of the edge, which determines the thickness of the link in the Sankey diagram.
    - "color": The color of the edge.

    Parameters:
    G (nx.Graph): A NetworkX graph with nodes and edges containing the necessary attributes.

    Returns:
    None

    Example:
    --------
    ```python
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.DiGraph()

    # Add nodes with color attributes
    G.add_node("A", color="blue")
    G.add_node("B", color="green")
    G.add_node("C", color="red")

    # Add edges with value and color attributes
    G.add_edge("A", "B", value=10, color="yellow")
    G.add_edge("B", "C", value=5, color="purple")

    generate_sankey(G)
    ```

    This example creates a simple directed graph with three nodes and two edges, and then generates a Sankey diagram.
    """
    sources, targets = [*zip(*G.edges)]
    nodes = list(G.nodes)

    get_node_indices = lambda node_list : [idx for node in node_list for idx, val in enumerate(nodes) if val == node]

    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 10,
            line = dict(color = "black", width = 0.5),
            label = [f"{node}" for node in G.nodes],
            align = "left",
            color = list(nx.get_node_attributes(G, "color").values())
        ),
        link = dict(
            source = get_node_indices(sources),
            target = get_node_indices(targets),
            value = list(nx.get_edge_attributes(G, "value").values()),
            color = list(nx.get_edge_attributes(G, "color").values())
        )
    )])

    fig.show()
