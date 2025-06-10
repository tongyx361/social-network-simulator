import asyncio
import logging
import os
import sqlite3
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import nest_asyncio
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from stqdm import stqdm

from oasis.social_platform.typing import ActionType
from sonetsim.utils.animation import (
    build_repost_network,
    get_action_data,
    get_follower_trend,
    get_sentiment_timeline,
    plot_action_animation,
    plot_follower_growth,
    plot_sentiment_timeline,
)

# c.f. https://sehmi-conscious.medium.com/got-that-asyncio-feeling-f1a7c37cab8b
nest_asyncio.apply()

logger = logging.getLogger(__name__)


def setup_debug_mode():
    logging.basicConfig(level=logging.DEBUG)
    import importlib

    from sonetsim import simulation
    from sonetsim.utils import models, temporal_graph

    importlib.reload(simulation)
    importlib.reload(temporal_graph)
    importlib.reload(models)


from sonetsim.simulation import (  # noqa: E402
    AGENT_INFO_FIELDS,
    ModelConfig,
    SimulationConfig,
    read_agent_info,
    simulate_twitter,
)
from sonetsim.utils.models import MODEL_PLATFORM2API_KEY_ENV_NAME  # noqa: E402


def disable_debug_mode():
    logging.basicConfig(level=logging.INFO)


def toggle_debug_mode():
    if st.session_state["debug_mode"]:
        setup_debug_mode()
    else:
        disable_debug_mode()


DEFAULT_AVAILABLE_ACTIONS = [
    ActionType.DO_NOTHING.value,
    ActionType.REPOST.value,
    ActionType.LIKE_POST.value,
    ActionType.FOLLOW.value,
]

STATE2DEFAULT: dict[str, Any] = {
    "simu_configs": None,
    "input_agent_info_df": pd.DataFrame(columns=AGENT_INFO_FIELDS),
}


def read_agent_info_file():
    file = st.session_state["init_agent_info_file"]
    try:
        st.session_state["input_agent_info_df"] = read_agent_info(file)
        if set(st.session_state["input_agent_info_df"].columns) != set(AGENT_INFO_FIELDS):
            raise ValueError("CSV file does not contain all required columns")
        st.session_state["base_agent_df"] = deepcopy(st.session_state["input_agent_info_df"])
        st.session_state["exp_agent_df"] = deepcopy(st.session_state["input_agent_info_df"])
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")


def simulation_col(agent_df_key: str, title: str):
    st.markdown(title)
    if agent_df_key not in st.session_state or st.session_state[agent_df_key].empty:
        st.session_state[agent_df_key] = deepcopy(st.session_state["input_agent_info_df"])
    # NOTE: `data_editor` edits a copy. c.f. https://github.com/streamlit/streamlit/blob/develop/lib/streamlit/elements/widgets/data_editor.py#L787
    st.session_state[agent_df_key] = st.data_editor(
        st.session_state[agent_df_key], use_container_width=True, hide_index=False, key=f"{agent_df_key}_editor"
    )  # NOTE: Setting `key` only stores changes in `st.session_state[key]`

    user_sel_col, info_col = st.columns(2)
    with user_sel_col:
        posting_user_name = st.selectbox(
            "Select User to Post",
            index=0,
            options=st.session_state[agent_df_key]["name"].tolist() if not st.session_state[agent_df_key].empty else [],
            key=f"{agent_df_key}_posting_user_sel",
        )

    # Show user information and posting interface after user selection
    if posting_user_name:
        # Get user information from session state dataframe
        posting_user_idx = st.session_state[agent_df_key][
            st.session_state[agent_df_key]["name"] == posting_user_name
        ].index[0]
        if not st.session_state[agent_df_key].loc[posting_user_idx].empty:
            with info_col:
                num_followers = (
                    int(st.session_state[agent_df_key].loc[posting_user_idx, "num_followers"])
                    if pd.notna(st.session_state[agent_df_key].loc[posting_user_idx, "num_followers"])
                    else 0
                )
                num_following = (
                    int(st.session_state[agent_df_key].loc[posting_user_idx, "num_following"])
                    if pd.notna(st.session_state[agent_df_key].loc[posting_user_idx, "num_following"])
                    else 0
                )
                st.markdown(f"Followers: *{num_followers}*")
                st.markdown(f"Following: *{num_following}*")
            with st.expander("##### ℹ️ Profile", expanded=True):
                st.text(st.session_state[agent_df_key].loc[posting_user_idx, "description"])

            if (
                isinstance(st.session_state[agent_df_key].loc[posting_user_idx, "previous_tweets"], list)
                and len(st.session_state[agent_df_key].loc[posting_user_idx, "previous_tweets"]) > 0
            ):
                print(
                    f"{agent_df_key=}, selected_user={posting_user_name}: "
                    f"{st.session_state[agent_df_key].loc[posting_user_idx, 'previous_tweets']}"
                )
                with st.container(height=200, border=True):
                    st.markdown("📋 Previous Posts")
                    for tweet in reversed(st.session_state[agent_df_key].loc[posting_user_idx, "previous_tweets"]):
                        st.info(tweet)
            else:
                st.text("##### 📭 No previous posts")

        post_content = st.text_area(
            label="Post Content",
            key=f"{agent_df_key}_post_content",
            placeholder="Input your base test case here...",
        )

        if st.button("Post", key=f"{agent_df_key}_post_button", type="secondary", icon="💬", use_container_width=True):
            if not post_content:
                st.error("❌ Please input post content")
                return
            if not st.session_state[agent_df_key].loc[posting_user_idx].empty:
                current_tweets = st.session_state[agent_df_key].loc[posting_user_idx, "previous_tweets"]
                new_tweets = current_tweets + [post_content]
                # Use .at for setting a single value in a DataFrame. This is more performant
                # and avoids pandas from trying to align the list with the DataFrame's index,
                # which would cause a ValueError.
                st.session_state[agent_df_key].at[posting_user_idx, "previous_tweets"] = new_tweets
                st.rerun()
            else:
                st.error(f"❌ User {posting_user_name} not found in the table")


# NOTE: Setting `key` also sets `st.session_state[key]`
if __name__ == "__main__":
    for state_key, default_value in STATE2DEFAULT.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value

    st.set_page_config(page_title="Social Network Simulator", layout="wide")
    st.title("Social Network Simulator")

    with st.sidebar:
        st.header("Simulation Settings")

        with st.expander("Model Backend", expanded=True, icon="🔑"):
            with st.form("api_key", border=False):
                st.selectbox(
                    "Model Platform",
                    index=[p for p in ModelPlatformType].index(ModelPlatformType.ZHIPU),
                    options=[platform.value for platform in ModelPlatformType],
                    key="model_platform",
                )
                st.selectbox(
                    "Model Type",
                    index=[m for m in ModelType].index(ModelType.GLM_4_FLASH),
                    options=[model_type.value for model_type in ModelType],
                    key="model_type",
                )
                st.text_input(
                    "API Key Env. Name",
                    value="ZHIPUAI_API_KEY",
                    key="api_key_env_name",
                    help=str({k.value: v for k, v in MODEL_PLATFORM2API_KEY_ENV_NAME.items()}),
                )
                api_key_value = st.text_input(
                    "API Key",
                    value=os.environ.get(st.session_state["api_key_env_name"]),
                    type="password",
                    key="api_key",
                )
                if api_key_value is not None:
                    os.environ[st.session_state["api_key_env_name"]] = api_key_value
                if st.form_submit_button("Validate"):
                    model = ModelFactory.create(
                        model_platform=ModelPlatformType(st.session_state["model_platform"]),
                        model_type=ModelType(st.session_state["model_type"]),
                    )
                    chat_agent = ChatAgent(model=model)
                    try:
                        response = chat_agent.step(input_message="Who are you?")
                    except Exception as e:
                        st.error(f"Error: {e}")
                    else:
                        st.success(str(response.msgs[0].content))

        st.multiselect(
            "Available Actions",
            options=[action.value for action in ActionType],
            default=DEFAULT_AVAILABLE_ACTIONS,
            key="available_actions",
        )

        st.toggle(
            "Debug Mode",
            on_change=toggle_debug_mode,
            key="debug_mode",
            value=os.environ.get("DEBUG", "false").lower() == "true",
        )

    st.header("🔮 Simulation")

    upload_col, download_col = st.columns(2, gap="large", vertical_alignment="bottom")
    with upload_col:
        st.file_uploader("Upload Base CSV", type="csv", key="init_agent_info_file", on_change=read_agent_info_file)
    with download_col:
        st.download_button(
            label="📥 Download Example CSV",
            data=open("data/agent_info/False_Business_0.csv", "rb").read(),
            file_name="False_Business_0.csv",
            mime="text/csv",
            help="Download this example CSV file to get started",
        )

    base_col, exp_col = st.columns(2)
    with base_col:
        simulation_col(agent_df_key="base_agent_df", title="### 📥 Base")
    with exp_col:
        simulation_col(agent_df_key="exp_agent_df", title="### 🧪 Experiment")

    simu_sel_col, num_timesteps_col = st.columns(2)
    with simu_sel_col:
        st.multiselect(
            "Select Simulations to Run",
            default=["Base", "Experiment"],
            options=["Base", "Experiment"],  # TODO: Dynamic options
            key="selected_simulation_ids",
        )

    with num_timesteps_col:
        st.number_input("Run for Timesteps", min_value=1, value=3, step=1, key="num_timesteps")

    if st.button("Simulate", use_container_width=True, type="primary", icon="🚀"):
        st.session_state["agent_info"] = {
            "Base": st.session_state["base_agent_df"],
            "Experiment": st.session_state["exp_agent_df"],
        }
        simu_db_home = Path("./data/simu_db")
        viz_home = Path("./data/visualization")

        # Generate unique UUID for this simulation run
        run_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters for brevity

        simu_configs: list[SimulationConfig] = []
        for simu_id in st.session_state["selected_simulation_ids"]:
            simu_config = SimulationConfig(
                model_config=ModelConfig(
                    model_platform=ModelPlatformType(st.session_state["model_platform"]),
                    model_type=ModelType(st.session_state["model_type"]),
                ),
                agent_info=st.session_state["agent_info"][simu_id],
                available_actions=st.session_state["available_actions"],
                db_path=simu_db_home / f"{simu_id}_{run_uuid}.db",
                visualization_home=viz_home / f"{simu_id}_{run_uuid}",
                num_timesteps=st.session_state["num_timesteps"],
                pbar=stqdm(desc=f"Simulating {simu_id}", total=st.session_state["num_timesteps"]),
            )
            simu_configs.append(simu_config)

        # Store the run UUID and simulation configs for later analysis
        st.session_state["current_run_uuid"] = run_uuid
        st.session_state["simu_configs"] = simu_configs

        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(asyncio.gather(*[simulate_twitter(config) for config in simu_configs]))
        except Exception as e:
            st.error(f"Error running simulations: {e}")
            raise e
        st.success("Simulations completed successfully")

    st.header("Analysis")

    # Frequency mapping for time-based analysis
    freq_map = {"hour": "H", "day": "D", "week": "W", "month": "M"}

    # For comparison analysis, we need both Base and Experiment databases
    # Get paths for both simulations for comparison visualizations
    db_path_base = None
    db_path_experiment = None

    if "current_run_uuid" in st.session_state:
        run_uuid = st.session_state["current_run_uuid"]
        db_path_base = Path(f"./data/simu_db/Base_{run_uuid}.db")
        db_path_experiment = Path(f"./data/simu_db/Experiment_{run_uuid}.db")
    else:
        # Fallback: look for any database files matching the patterns
        simu_db_home = Path("./data/simu_db")
        base_files = list(simu_db_home.glob("Base_*.db"))
        exp_files = list(simu_db_home.glob("Experiment_*.db"))

        if base_files:
            db_path_base = max(base_files, key=lambda p: p.stat().st_mtime)
        else:
            db_path_base = Path("./data/simu_db/Base.db")

        if exp_files:
            db_path_experiment = max(exp_files, key=lambda p: p.stat().st_mtime)
        else:
            db_path_experiment = Path("./data/simu_db/Experiment.db")

    # Check if comparison databases exist for visualization
    if not db_path_base.exists() or not db_path_experiment.exists():
        st.warning(
            "Both Base and Experiment databases are needed for comparison analysis. Some visualizations may not work."
        )
        # Create dummy connections for non-existent databases
        conn_base = sqlite3.connect(":memory:") if not db_path_base.exists() else sqlite3.connect(db_path_base)
        conn_experiment = (
            sqlite3.connect(":memory:") if not db_path_experiment.exists() else sqlite3.connect(db_path_experiment)
        )
    else:
        conn_base = sqlite3.connect(db_path_base)
        conn_experiment = sqlite3.connect(db_path_experiment)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Action Count Animation", "📈 Follower Trend", "🌐 Repost Network", "💬 Comment Sentiment Timeline"]
    )

    with tab1:
        st.markdown("#### Action Count Animation")
        selected_actions = st.multiselect(
            "Select Actions",
            ["like", "dislike", "comment", "follow", "repost"],
            default=["like", "comment", "follow", "repost"],
        )

        df_action = get_action_data(conn_base, conn_experiment, selected_actions)
        if df_action is not None:
            st.markdown("#### 📊 Total Action Counts")
            summary = (
                df_action.groupby(["source", "action"])
                .size()
                .reset_index(name="count")
                .pivot(index="action", columns="source", values="count")
                .fillna(0)
                .astype(int)
            )
            st.dataframe(summary, use_container_width=True)
            fig = plot_action_animation(df_action)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No action data found for selected actions.")

    with tab2:
        st.markdown("#### Follower Growth Over Time")
        top_k = st.slider("Number of Top Users", 1, 20, 5)
        trend = get_follower_trend(conn_base, conn_experiment, top_k=top_k)
        fig = plot_follower_growth(trend)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### User Repost Network")
        path = build_repost_network(conn_base, conn_experiment)
        components.html(open(path).read(), height=600)

    with tab4:
        st.markdown("#### Sentiment of Comments Over Time")
        sentiment = get_sentiment_timeline(conn_base, conn_experiment)
        fig = plot_sentiment_timeline(sentiment)
        st.plotly_chart(fig, use_container_width=True)
