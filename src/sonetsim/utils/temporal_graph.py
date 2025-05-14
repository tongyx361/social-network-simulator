# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from sonetsim.utils.graph import plot_graph_like_tree

logger = logging.getLogger(__name__)


@dataclass
class TemporalEdge:
    source: int
    dest: int
    time: int


class TemporalGraph:
    """
    TODO: Maybe we can make it more general beyond repost.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = db_path  # Path to the db file obtained after simulation

    def __enter__(self):
        # Connect to the SQLite database when entering the context
        assert self.db_path is not None, "Database path not provided"
        self.conn = sqlite3.connect(self.db_path)
        if self.conn:
            self.df = self.load_data()
            self.G = self.build_graph(self.df)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the database connection when exiting the context
        if self.conn:
            self.conn.close()

    def load_data(self) -> pd.DataFrame:
        # Check if a connection exists (it should if used within a 'with' statement)
        if not self.conn:
            raise RuntimeError("Database connection not established. Use 'with' statement.")

        # Execute SQL query and load the results into a DataFrame
        query = "SELECT * FROM post"
        df = pd.read_sql(query, self.conn)
        return df

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        edges: list[TemporalEdge] = []

        for i in range(len(df)):
            row = df.loc[i]
            edge_data = {
                "source": row["original_post_id"],
                "dest": row["post_id"],
                "time": row["created_at"],
            }
            if not all(v for v in edge_data.values()):
                continue
            edges.append(
                TemporalEdge(source=int(edge_data["source"]), dest=int(edge_data["dest"]), time=int(edge_data["time"]))
            )

        G: nx.DiGraph = nx.DiGraph()

        # Extract edges from the data and add them to the graph
        for edge in edges:
            if edge.source not in G:
                G.add_node(edge.source, time=edge.time)
            if edge.dest not in G:
                G.add_node(edge.dest, time=edge.time)
            G.add_edge(edge.source, edge.dest)

        return G

    def get_subgraph(self, by_time: int | None = None):
        assert self.G is not None, "Graph not built"
        if by_time is None:
            return self.G

        filtered_nodes = []
        for node, attr in self.G.nodes(data=True):
            if attr["time"] <= by_time:
                filtered_nodes.append(node)
        subG = self.G.subgraph(filtered_nodes)

        return subG

    def plot(self, source: int, by_time: int | None = None):
        subG = self.get_subgraph(by_time)
        plot_graph_like_tree(subG, source)
