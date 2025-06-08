import asyncio
import logging
import os
import sqlite3
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

    with st.expander("🔮 Simulation Settings: Compare Base vs. Experiment", expanded=True):
        st.markdown(
            """
            Upload the agent information for your **Base** simulation on the left.
            On the right, you can edit the **Experiment** configuration derived from the base.
            This allows you to easily compare and adjust settings between the two scenarios.
            """
        )

        base_agent_info_file = st.file_uploader("Upload Base CSV", type="csv")
        just_read_df = pd.read_csv(base_agent_info_file, index_col=0) if base_agent_info_file else None
        base_agent_info_df = just_read_df if just_read_df is not None else pd.DataFrame(columns=AGENT_INFO_FIELDS)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📥 Base Agent Information**")
            if not set(base_agent_info_df.columns) == set(AGENT_INFO_FIELDS):
                st.error("The uploaded CSV file does not contain the required columns.")
            else:
                edited_base_df = st.data_editor(
                    base_agent_info_df,
                    num_rows="dynamic",
                    key="edited_base_agent_info_df_changes",
                    use_container_width=True,
                    hide_index=False,
                )
                base_test_prompt = st.text_area(
                    label="📝 Base Scenario Test Prompt",
                    placeholder=(
                        "Describe what you'd like to test with the base agents (you can use emojis like 🚀😊🔍)..."
                    ),
                    key="base_test_prompt",
                )
                st.caption(
                    "This input will be used to define test conditions for the base scenario. Feel free to add emojis!"
                )
                selected_base_rows = st.session_state.get("edited_base_agent_info_df_changes", {}).get(
                    "edited_rows", []
                )
                if selected_base_rows:
                    for idx in selected_base_rows:
                        edited_base_df.at[idx, "previous_tweets"] = base_test_prompt
                    st.success(f"✅ Applied prompt to selected base row(s): {selected_base_rows}")
                else:
                    st.info("Please select at least one row in the table to apply the test prompt.")
                st.session_state["agent_info"]["Base"] = edited_base_df

        with col2:
            st.markdown("**🧪 Experiment Agent Information**")
            if "agent_info" not in st.session_state or "Base" not in st.session_state["agent_info"]:
                st.info("Please upload and edit the Base Agent Information first.")
                st.session_state["agent_info"]["Experiment"] = pd.DataFrame(columns=AGENT_INFO_FIELDS)
            else:
                edited_exp_df = st.data_editor(
                    st.session_state["agent_info"]["Base"],
                    num_rows="dynamic",
                    key="edited_exp_agent_info_df_changes",
                    use_container_width=True,
                    hide_index=False,
                )

                experiment_test_prompt = st.text_area(
                    label="📝 Experiment Scenario Test Prompt",
                    placeholder="Describe your experimental test case here (emojis supported: 🤖💡⚔️)...",
                    key="experiment_test_prompt",
                )
                st.caption(
                    "This input defines the modifications or goals for the experimental setup. "
                    "Express creatively with emojis!"
                )
                selected_exp_rows = st.session_state.get("edited_exp_agent_info_df_changes", {}).get("edited_rows", [])

                if selected_exp_rows:
                    for idx in selected_exp_rows:
                        edited_exp_df.at[idx, "previous_tweets"] = experiment_test_prompt
                    st.success(f"✅ Applied prompt to selected experiment row(s): {selected_exp_rows}")
                else:
                    st.info("Please select at least one row in the table to apply the test prompt.")
                st.session_state["agent_info"]["Experiment"] = edited_exp_df

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

    db_path_base = Path("./data/simu_db/Base.db")
    db_path_experiment = Path("./data/simu_db/Experiment.db")
    if not db_path_base.exists():
        st.error("Database Base does not exist. Please run simulations first.")
        st.stop()
    if not db_path_experiment.exists():
        st.error("Database Experiment does not exist. Please run simulations first.")
        st.stop()
    conn_base = sqlite3.connect(db_path_base)
    conn_experiment = sqlite3.connect(db_path_experiment)

    st.subheader("📈 Behavior Analysis Dashboard")

    chart_type = st.radio(
        "Select Analysis Type",
        [
            "📊 Action Count Animation",
            "📈 Follower Trend",
            "🌐 Repost Network",
            "💬 Comment Sentiment Timeline",
        ],
    )

    if chart_type == "📊 Action Count Animation":
        st.markdown("#### Action Count Animation")
        selected_actions = st.multiselect(
            "Select Actions",
            ["like", "dislike", "comment", "follow", "repost"],
            default=["like", "comment"],
        )

        df_action = get_action_data(conn_base, conn_experiment, selected_actions)
        if df_action is not None:
            fig = plot_action_animation(df_action)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No action data found for selected actions.")

    elif chart_type == "📈 Follower Trend":
        st.markdown("#### Follower Growth Over Time")
        top_k = st.slider("Number of Top Users", 1, 20, 5)
        trend = get_follower_trend(conn_base, conn_experiment, top_k=top_k)
        fig = plot_follower_growth(trend)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "🌐 Repost Network":
        st.markdown("#### User Repost Network")
        path = build_repost_network(conn_base, conn_experiment)
        components.html(open(path).read(), height=600)

    elif chart_type == "💬 Comment Sentiment Timeline":
        st.markdown("#### Sentiment of Comments Over Time")
        sentiment = get_sentiment_timeline(conn_base, conn_experiment)
        fig = plot_sentiment_timeline(sentiment)
        st.plotly_chart(fig, use_container_width=True)
