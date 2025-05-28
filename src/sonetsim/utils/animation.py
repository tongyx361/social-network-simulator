import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import pyvis.network as net
import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob


def show_action_animation(conn):
    freq_map = {"day": "D", "week": "W", "month": "M", "hour": "H"}

    # 时间范围选择
    with st.expander("Filter Options", expanded=False):
        action_types = ["like", "dislike", "comment", "follow", "repost"]
        selected_actions = st.multiselect("Select actions to visualize", options=action_types, default=action_types)
        time_bin_input = st.selectbox("Select time bin", options=["day", "week", "month", "hour"], index=0)
        freq = freq_map.get(time_bin_input.lower(), "D")

    # 将各类行为统一成一个DataFrame用于动画
    all_actions = []

    if "like" in selected_actions:
        df = pd.read_sql_query("SELECT user_id, created_at FROM like", conn)
        df["action"] = "like"
        all_actions.append(df)

    if "dislike" in selected_actions:
        df = pd.read_sql_query("SELECT user_id, created_at FROM dislike", conn)
        df["action"] = "dislike"
        all_actions.append(df)

    if "comment" in selected_actions:
        df = pd.read_sql_query("SELECT user_id, created_at FROM comment", conn)
        df["action"] = "comment"
        all_actions.append(df)

    if "follow" in selected_actions:
        df = pd.read_sql_query("SELECT follower_id AS user_id, created_at FROM follow", conn)
        df["action"] = "follow"
        all_actions.append(df)

    if "repost" in selected_actions:
        df = pd.read_sql_query("SELECT user_id, created_at FROM post WHERE original_post_id IS NOT NULL", conn)
        df["action"] = "repost"
        all_actions.append(df)

    if all_actions:
        df_all = pd.concat(all_actions)
        df_all["created_at"] = pd.to_datetime(df_all["created_at"])
        df_all["time_bin"] = df_all["created_at"].dt.to_period(freq).dt.to_timestamp()

        # Group by time + action
        agg_df = df_all.groupby(["time_bin", "action"]).size().reset_index(name="count")

        # 动态柱状图
        fig = px.bar(
            agg_df,
            x="action",
            y="count",
            color="action",
            animation_frame=agg_df["time_bin"].astype(str),
            range_y=[0, agg_df["count"].max() * 1.2],
            title="User Actions Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one action type to visualize.")


def show_follower_trend(conn):
    df = pd.read_sql_query("SELECT followee_id, created_at FROM follow", conn)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["day"] = df["created_at"].dt.to_period("D").dt.to_timestamp()

    trend = df.groupby(["day", "followee_id"]).size().reset_index(name="new_followers")
    trend["cumulative"] = trend.groupby("followee_id")["new_followers"].cumsum()

    top_users = trend.groupby("followee_id")["cumulative"].max().nlargest(5).index
    trend = trend[trend["followee_id"].isin(top_users)]

    fig = px.line(trend, x="day", y="cumulative", color="followee_id", title="Top Users' Follower Growth Over Time")
    st.plotly_chart(fig, use_container_width=True)


def show_post_popularity_flow(conn):
    post_id = st.number_input("Enter Post ID", min_value=0)

    df = pd.read_sql_query(
        """
        SELECT post_id, user_id, original_post_id FROM post
        WHERE original_post_id IS NOT NULL
    """,
        conn,
    )

    # 过滤出该 post 的传播链
    G = nx.DiGraph()
    stack = [post_id]

    while stack:
        current = stack.pop()
        children = df[df["original_post_id"] == current]
        for _, row in children.iterrows():
            G.add_edge(current, row["post_id"])
            stack.append(row["post_id"])

    if G.number_of_nodes() == 0:
        st.warning("This post has no reposts.")
        return

    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", ax=ax)
    st.pyplot(fig)


def show_repost_network(conn):
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
    g.save_graph("repost_net.html")
    components.html(open("repost_net.html").read(), height=600)


def show_comment_sentiment_timeline(conn):
    df = pd.read_sql_query("SELECT comment_id, content, created_at FROM comment", conn)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["day"] = df["created_at"].dt.to_period("D").dt.to_timestamp()

    df["polarity"] = df["content"].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)

    sentiment = df.groupby("day")["polarity"].mean().reset_index()
    fig = px.line(sentiment, x="day", y="polarity", title="Average Comment Sentiment Over Time")
    st.plotly_chart(fig, use_container_width=True)
