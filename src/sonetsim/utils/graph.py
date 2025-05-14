import logging

import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)


def calc_max_depth(G: nx.DiGraph, source: int) -> int:
    dfs_tree = nx.dfs_tree(G, source=source)
    max_depth = max(nx.single_source_shortest_path_length(dfs_tree, source=source).values())
    return max_depth


def calc_max_breadth(G: nx.DiGraph, source: int, max_depth: int | None = None) -> int:
    if max_depth is None:
        max_depth = calc_max_depth(G, source)
    max_breadth = 0
    layer_breadths = [1]
    for depth_limit in range(1, max_depth + 1):
        breadth = len(list(nx.bfs_tree(G, source=source, depth_limit=depth_limit).nodes())) - sum(layer_breadths)
        layer_breadths.append(breadth)
        if breadth > max_breadth:
            max_breadth = breadth
    return max_breadth


def hierarchy_pos(
    G: nx.DiGraph, root=None, width: float = 1.0, vert_gap: float = 0.2, vert_loc: float = 0, xcenter: float = 0.5
):
    """Compute the positions of all nodes in the tree starting from a given root node position"""
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(
    G: nx.DiGraph,
    root: int,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0,
    xcenter: float = 0.5,
    pos: dict[int, tuple[float, float]] | None = None,
    parent: int | None = None,
    parsed: set[int] | None = None,
):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    if parsed is None:
        parsed = {root}
    else:
        parsed.add(root)

    neighbors = list(G.neighbors(root))
    # logger.debug(f"{neighbors=}")
    if parent in neighbors:
        neighbors.remove(parent)

    if len(neighbors) != 0:
        dx = width / len(neighbors)  # Horizontal space allocated for each node
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos = _hierarchy_pos(
                G,
                root=neighbor,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=nextx,
                pos=pos,
                parent=root,
                parsed=parsed,
            )
    return pos


def plot_graph_like_tree(G: nx.DiGraph, root: int) -> None:
    pos = hierarchy_pos(G, root)
    logger.debug(f"{pos=}")

    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrows=True,
    )
    plt.show()
