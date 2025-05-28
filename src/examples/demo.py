import asyncio
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import nest_asyncio
import pandas as pd
import streamlit as st
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from stqdm import stqdm

from oasis.social_platform.typing import ActionType
from sonetsim.utils.animation import (
    show_action_animation,
    show_comment_sentiment_timeline,
    show_follower_trend,
    show_post_popularity_flow,
    show_repost_network,
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


from sonetsim.simulation import ModelConfig, SimulationConfig, simulate_twitter  # noqa: E402
from sonetsim.utils.models import MODEL_PLATFORM2API_KEY_ENV_NAME  # noqa: E402


def disable_debug_mode():
    logging.basicConfig(level=logging.INFO)


def toggle_debug_mode():
    if st.session_state["debug_mode"]:
        setup_debug_mode()
    else:
        disable_debug_mode()


AGENT_INFO_FIELDS = [
    "user_id",
    "name",
    "username",
    "following_agentid_list",
    "previous_tweets",
    "user_char",
    "description",
]


DEFAULT_AVAILABLE_ACTIONS = [
    ActionType.DO_NOTHING.value,
    ActionType.REPOST.value,
    ActionType.LIKE_POST.value,
    ActionType.FOLLOW.value,
]

STATE2DEFAULT: dict[str, Any] = {"simu_configs": None, "agent_info": {}}

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

    with st.expander("Simulation", expanded=True, icon="🔮"):
        # The uploaded `base_agent_info_file` is immediately used to build the `base_agent_info_df`
        # without the need to put it in `st.session_state`
        base_agent_info_file = st.file_uploader("Upload Base Agent Information", type="csv")
        just_read_df = pd.read_csv(base_agent_info_file, index_col=0) if base_agent_info_file is not None else None
        # logger.debug(f"{just_read_df=}")
        base_agent_info_df = just_read_df if just_read_df is not None else pd.DataFrame(columns=AGENT_INFO_FIELDS)
        assert set(base_agent_info_df.columns) == set(AGENT_INFO_FIELDS), (
            f"{set(base_agent_info_df.columns)=} != {set(AGENT_INFO_FIELDS)=}"
        )

        st.write("Base Agent Information")
        # `data_editor(key="data_editor")` only saves the changes to `st.session_state["data_editor"]`
        st.session_state["agent_info"]["Base"] = st.data_editor(
            base_agent_info_df, num_rows="dynamic", key="edited_base_agent_info_df_changes"
        )
        # TODO: Multiple simulations
        st.write("Agent Information for Comparison (Refreshing if base changes)")
        st.session_state["agent_info"]["Experiment"] = st.data_editor(
            st.session_state["agent_info"]["Base"], num_rows="dynamic", key="edited_exp_agent_info_df_changes"
        )

        st.multiselect(
            "Select Simulations to Run",
            default=["Base", "Experiment"],
            options=["Base", "Experiment"],  # TODO: Dynamic options
            key="selected_simulation_ids",
        )

        # Prepare simulation configurations based on UI selections

        st.number_input("Run for Timesteps", min_value=1, value=3, step=1, key="num_timesteps")
        if st.button("Run Selected Simulations"):
            simu_db_home = Path("./data/simu_db")
            viz_home = Path("./data/visualization")

            simu_configs: list[SimulationConfig] = []
            for simu_id in st.session_state["selected_simulation_ids"]:
                simu_config = SimulationConfig(
                    model_config=ModelConfig(
                        model_platform=ModelPlatformType(st.session_state["model_platform"]),
                        model_type=ModelType(st.session_state["model_type"]),
                    ),
                    agent_info=st.session_state["agent_info"][simu_id],
                    available_actions=st.session_state["available_actions"],
                    db_path=simu_db_home / f"{simu_id}.db",
                    visualization_home=viz_home / simu_id,
                    num_timesteps=st.session_state["num_timesteps"],
                    pbar=stqdm(desc=f"Simulating {simu_id}", total=st.session_state["num_timesteps"]),
                )
                simu_configs.append(simu_config)

            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(asyncio.gather(*[simulate_twitter(config) for config in simu_configs]))
            except Exception as e:
                st.error(f"Error running simulations: {e}")
                raise e
            st.success("Simulations completed successfully")

    st.header("Analysis")

    simu_id_to_analyze = st.selectbox(
        "Select Simulation to Analyze",
        options=["Base", "Experiment"],
        index=0,
        key="simu_id_to_analyze",
    )

    st.subheader("📈 Behavior Analysis Dashboard")

    chart_type = st.radio(
        "Select Analysis Type",
        [
            "📊 Action Count Animation",
            "📈 Follower Trend",
            "🔥 Post Popularity Flow",
            "🌐 Repost Network",
            "💬 Comment Sentiment Timeline",
        ],
    )

    db_path = Path(f"./data/simu_db/{simu_id_to_analyze}.db")
    if not db_path.exists():
        st.error(f"Database {db_path} does not exist. Please run simulations first.")
        st.stop()
    conn = sqlite3.connect(db_path)

    if chart_type == "📊 Action Count Animation":
        show_action_animation(conn)

    elif chart_type == "📈 Follower Trend":
        show_follower_trend(conn)

    elif chart_type == "🔥 Post Popularity Flow":
        show_post_popularity_flow(conn)

    elif chart_type == "🌐 Repost Network":
        show_repost_network(conn)

    elif chart_type == "💬 Comment Sentiment Timeline":
        show_comment_sentiment_timeline(conn)
