import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
from pyvis.network import Network
from textblob import TextBlob


def get_action_data(conn_base, conn_experiment, selected_actions):
    action_sql_map = {
        "like": "SELECT user_id, created_at FROM like",
        "dislike": "SELECT user_id, created_at FROM dislike",
        "comment": "SELECT user_id, created_at FROM comment",
        "follow": "SELECT follower_id AS user_id, created_at FROM follow",
        "repost": "SELECT user_id, created_at FROM post WHERE original_post_id IS NOT NULL",
    }

    def extract_actions(conn, source_label):
        actions = []
        for action in selected_actions:
            if action in action_sql_map:
                df = pd.read_sql_query(action_sql_map[action], conn)
                df["action"] = action
                df["source"] = source_label  # Add source label
                actions.append(df)
        return actions

    base_actions = extract_actions(conn_base, "Base")
    exp_actions = extract_actions(conn_experiment, "Experiment")
    all_actions = base_actions + exp_actions
    if not all_actions:
        return None

    df_all = pd.concat(all_actions, ignore_index=True)
    df_all["created_at"] = pd.to_datetime(df_all["created_at"])
    df_all["time_bin"] = df_all["created_at"]

    return df_all


def plot_action_animation(df_all):
    agg_df = df_all.groupby(["time_bin", "action", "source"]).size().reset_index(name="count")

    fig = px.bar(
        agg_df,
        x="action",
        y="count",
        color="source",
        animation_frame=agg_df["time_bin"].astype(str),
        barmode="group",
        range_y=[0, agg_df["count"].max() * 1.2],
        title="User Actions Over Time (Base vs. Experiment)",
    )
    return fig


def get_follower_trend(conn_base, conn_experiment, top_k=5):
    def fetch_trend(conn, source_label):
        df = pd.read_sql_query("SELECT followee_id, created_at FROM follow", conn)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["day"] = df["created_at"]

        trend = df.groupby(["day", "followee_id"]).size().reset_index(name="new_followers")
        trend["cumulative"] = trend.groupby("followee_id")["new_followers"].cumsum()
        trend["source"] = source_label
        return trend

    base_trend = fetch_trend(conn_base, "Base")
    exp_trend = fetch_trend(conn_experiment, "Experiment")
    all_trend = pd.concat([base_trend, exp_trend], ignore_index=True)
    top_users = all_trend.groupby("followee_id")["cumulative"].max().nlargest(top_k).index
    all_trend = all_trend[all_trend["followee_id"].isin(top_users)]
    return all_trend


def plot_follower_growth(trend):
    fig = px.line(trend, x="day", y="cumulative", color="followee_id", title="Top Users' Follower Growth Over Time")
    return fig


def build_post_graph(conn_base, conn_experiment, root_post_id_base, root_post_id_experiment):
    def fetch_post_edges(conn, root_post_id, source_label):
        df = pd.read_sql_query(
            "SELECT post_id, user_id, original_post_id FROM post WHERE original_post_id IS NOT NULL",
            conn,
        )
        G_sub = nx.DiGraph()
        stack = [root_post_id]
        while stack:
            current = stack.pop()
            children = df[df["original_post_id"] == current]
            for _, row in children.iterrows():
                G_sub.add_edge(f"{source_label}_{current}", f"{source_label}_{row['post_id']}", source=source_label)
                stack.append(row["post_id"])
        return G_sub

    G_base = fetch_post_edges(conn_base, root_post_id_base, "Base")
    G_exp = fetch_post_edges(conn_experiment, root_post_id_experiment, "Experiment")
    G_combined = nx.compose(G_base, G_exp)
    return G_combined


def plot_post_popularity_flow(G):
    if G.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", ax=ax)
    return fig


def build_repost_network(conn_base, conn_experiment, save_path="repost_net.html"):
    def get_edges(conn, label):
        df = pd.read_sql_query(
            """
            SELECT p1.user_id as source_user, p2.user_id as target_user
            FROM post p1
            JOIN post p2 ON p1.post_id = p2.original_post_id
            WHERE p2.original_post_id IS NOT NULL
            """,
            conn,
        )
        df["source_user"] = df["source_user"].astype(str).apply(lambda x: f"{label}_{x}")
        df["target_user"] = df["target_user"].astype(str).apply(lambda x: f"{label}_{x}")
        return df

    df_base = get_edges(conn_base, "Base")
    df_exp = get_edges(conn_experiment, "Experiment")

    g = Network(height="600px", width="100%", notebook=False, directed=True)

    for _, row in df_base.iterrows():
        g.add_node(row["source_user"], color="blue")
        g.add_node(row["target_user"], color="blue")
        g.add_edge(row["source_user"], row["target_user"], color="blue")

    for _, row in df_exp.iterrows():
        g.add_node(row["source_user"], color="orange")
        g.add_node(row["target_user"], color="orange")
        g.add_edge(row["source_user"], row["target_user"], color="orange")

    g.set_options("""
        var options = {
          "nodes": { "shape": "dot", "size": 8 },
          "physics": { "barnesHut": { "gravitationalConstant": -30000 } }
        }
    """)
    g.save_graph(save_path)
    return save_path


def get_sentiment_timeline(conn_base, conn_experiment):
    def extract_sentiment(conn, label):
        df = pd.read_sql_query("SELECT comment_id, content, created_at FROM comment", conn)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["day"] = df["created_at"].dt.to_period("D").dt.to_timestamp()
        df["polarity"] = df["content"].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)
        sentiment = df.groupby("day")["polarity"].mean().reset_index()
        sentiment["source"] = label
        return sentiment

    base_sentiment = extract_sentiment(conn_base, "Base")
    exp_sentiment = extract_sentiment(conn_experiment, "Experiment")

    combined = pd.concat([base_sentiment, exp_sentiment])
    return combined


def plot_sentiment_timeline(sentiment):
    fig = px.line(sentiment, x="day", y="polarity", title="Average Comment Sentiment Over Time")
    return fig
