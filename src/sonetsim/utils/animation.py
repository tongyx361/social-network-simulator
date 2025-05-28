import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import pyvis.network as net
from textblob import TextBlob


def get_action_data(conn, selected_actions, freq="D"):
    action_sql_map = {
        "like": "SELECT user_id, created_at FROM like",
        "dislike": "SELECT user_id, created_at FROM dislike",
        "comment": "SELECT user_id, created_at FROM comment",
        "follow": "SELECT follower_id AS user_id, created_at FROM follow",
        "repost": "SELECT user_id, created_at FROM post WHERE original_post_id IS NOT NULL",
    }

    all_actions = []
    for action in selected_actions:
        if action in action_sql_map:
            df = pd.read_sql_query(action_sql_map[action], conn)
            df["action"] = action
            all_actions.append(df)

    if not all_actions:
        return None

    df_all = pd.concat(all_actions)
    df_all["created_at"] = pd.to_datetime(df_all["created_at"])
    df_all["time_bin"] = df_all["created_at"].dt.to_period(freq).dt.to_timestamp()

    return df_all


def plot_action_animation(df_all):
    agg_df = df_all.groupby(["time_bin", "action"]).size().reset_index(name="count")

    fig = px.bar(
        agg_df,
        x="action",
        y="count",
        color="action",
        animation_frame=agg_df["time_bin"].astype(str),
        range_y=[0, agg_df["count"].max() * 1.2],
        title="User Actions Over Time",
    )
    return fig


def get_follower_trend(conn, top_k=5):
    df = pd.read_sql_query("SELECT followee_id, created_at FROM follow", conn)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["day"] = df["created_at"].dt.to_period("D").dt.to_timestamp()

    trend = df.groupby(["day", "followee_id"]).size().reset_index(name="new_followers")
    trend["cumulative"] = trend.groupby("followee_id")["new_followers"].cumsum()

    top_users = trend.groupby("followee_id")["cumulative"].max().nlargest(top_k).index
    trend = trend[trend["followee_id"].isin(top_users)]

    return trend


def plot_follower_growth(trend):
    fig = px.line(trend, x="day", y="cumulative", color="followee_id", title="Top Users' Follower Growth Over Time")
    return fig


def build_post_graph(conn, root_post_id):
    df = pd.read_sql_query(
        "SELECT post_id, user_id, original_post_id FROM post WHERE original_post_id IS NOT NULL", conn
    )

    G = nx.DiGraph()
    stack = [root_post_id]

    while stack:
        current = stack.pop()
        children = df[df["original_post_id"] == current]
        for _, row in children.iterrows():
            G.add_edge(current, row["post_id"])
            stack.append(row["post_id"])

    return G


def plot_post_popularity_flow(G):
    if G.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", ax=ax)
    return fig


def build_repost_network(conn, save_path="repost_net.html"):
    df = pd.read_sql_query(
        """
        SELECT p1.user_id as source_user, p2.user_id as target_user
        FROM post p1
        JOIN post p2 ON p1.post_id = p2.original_post_id
        WHERE p2.original_post_id IS NOT NULL
        """,
        conn,
    )

    g = net.Network(height="600px", width="100%", notebook=False, directed=True)
    for _, row in df.iterrows():
        g.add_node(str(row["source_user"]))
        g.add_node(str(row["target_user"]))
        g.add_edge(str(row["source_user"]), str(row["target_user"]))

    g.set_options("""
        var options = {
          "nodes": { "shape": "dot", "size": 8 },
          "physics": { "barnesHut": { "gravitationalConstant": -30000 } }
        }
    """)
    g.save_graph(save_path)
    return save_path


def get_sentiment_timeline(conn):
    df = pd.read_sql_query("SELECT comment_id, content, created_at FROM comment", conn)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["day"] = df["created_at"].dt.to_period("D").dt.to_timestamp()
    df["polarity"] = df["content"].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)

    sentiment = df.groupby("day")["polarity"].mean().reset_index()
    return sentiment


def plot_sentiment_timeline(sentiment):
    fig = px.line(sentiment, x="day", y="polarity", title="Average Comment Sentiment Over Time")
    return fig
