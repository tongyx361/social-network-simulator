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
    build_post_graph,
    build_repost_network,
    get_action_data,
    get_follower_trend,
    get_sentiment_timeline,
    plot_action_animation,
    plot_follower_growth,
    plot_post_popularity_flow,
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


def render_agent_info_editor(
    scenario_name: str,
    dataframe: pd.DataFrame,
    data_editor_key: str,
    text_area_key: str,
    title: str,
    prompt_label: str,
    prompt_placeholder: str,
    prompt_caption: str,
    prerequisite_check_func=None,
    prerequisite_message: str = "",
):
    """
    Render a common agent information editor UI component.

    Args:
        scenario_name: Name of the scenario (e.g., "Base", "Experiment")
        dataframe: DataFrame to edit
        data_editor_key: Key for the data editor component
        text_area_key: Key for the text area component
        title: Title to display for this section
        prompt_label: Label for the test prompt text area
        prompt_placeholder: Placeholder text for the test prompt
        prompt_caption: Caption/help text for the test prompt
        prerequisite_check_func: Optional function to check prerequisites
        prerequisite_message: Message to show when prerequisites aren't met

    Returns:
        Edited DataFrame or None if prerequisites not met
    """
    st.markdown(f"**{title}**")

    # Check prerequisites if provided
    if prerequisite_check_func and not prerequisite_check_func():
        st.info(prerequisite_message)
        st.session_state["agent_info"][scenario_name] = pd.DataFrame(columns=AGENT_INFO_FIELDS)
        return None

    # Render data editor
    edited_df = st.data_editor(
        dataframe,
        num_rows="dynamic",
        key=data_editor_key,
        use_container_width=True,
        hide_index=False,
    )

    # Render test prompt text area
    test_prompt = st.text_area(
        label=prompt_label,
        placeholder=prompt_placeholder,
        key=text_area_key,
    )
    st.caption(prompt_caption)

    # Handle selected rows and apply prompt
    selected_rows = st.session_state.get(data_editor_key, {}).get("edited_rows", [])
    if selected_rows:
        for idx in selected_rows:
            edited_df.at[idx, "previous_tweets"] = test_prompt
        st.success(f"✅ Applied prompt to selected {scenario_name.lower()} row(s): {selected_rows}")
    else:
        st.info("Please select at least one row in the table to apply the test prompt.")

    # Store in session state
    st.session_state["agent_info"][scenario_name] = edited_df
    return edited_df


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
        base_agent_info_df = (
            read_agent_info(base_agent_info_file) if base_agent_info_file else pd.DataFrame(columns=AGENT_INFO_FIELDS)
        )

        col1, col2 = st.columns(2)

        with col1:
            # Handle base agent information with validation
            if not set(base_agent_info_df.columns) == set(AGENT_INFO_FIELDS):
                st.markdown("**📥 Base Agent Information**")
                st.error("The uploaded CSV file does not contain the required columns.")
            else:
                render_agent_info_editor(
                    scenario_name="Base",
                    dataframe=base_agent_info_df,
                    data_editor_key="edited_base_agent_info_df_changes",
                    text_area_key="base_test_prompt",
                    title="📥 Base Agent Information",
                    prompt_label="📝 Base Scenario Test Prompt",
                    prompt_placeholder="Describe what you'd like to test with the base agents "
                    "(you can use emojis like 🚀😊🔍)...",
                    prompt_caption="This input will be used to define test conditions for the base scenario. "
                    "Feel free to add emojis!",
                )

        with col2:
            # Define prerequisite check function for experiment
            def check_base_prerequisites():
                return "agent_info" in st.session_state and "Base" in st.session_state["agent_info"]

            render_agent_info_editor(
                scenario_name="Experiment",
                dataframe=st.session_state["agent_info"].get("Base", pd.DataFrame(columns=AGENT_INFO_FIELDS)),
                data_editor_key="edited_exp_agent_info_df_changes",
                text_area_key="experiment_test_prompt",
                title="🧪 Experiment Agent Information",
                prompt_label="📝 Experiment Scenario Test Prompt",
                prompt_placeholder="Describe your experimental test case here (emojis supported: 🤖💡⚔️)...",
                prompt_caption="This input defines the modifications or goals for the experimental setup. "
                "Express creatively with emojis!",
                prerequisite_check_func=check_base_prerequisites,
                prerequisite_message="Please upload and edit the Base Agent Information first.",
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

    freq_map = {"hour": "H", "day": "D", "week": "W", "month": "M"}

    simu_id_to_analyze = st.selectbox(
        "Select Simulation to Analyze",
        options=["Base", "Experiment"],
        index=0,
        key="simu_id_to_analyze",
    )
    db_path = Path(f"./data/simu_db/{simu_id_to_analyze}.db")
    if not db_path.exists():
        st.error(f"Database {db_path} does not exist. Please run simulations first.")
        st.stop()
    conn = sqlite3.connect(db_path)

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

    if chart_type == "📊 Action Count Animation":
        st.markdown("#### Action Count Animation")
        selected_actions = st.multiselect(
            "Select Actions",
            ["like", "dislike", "comment", "follow", "repost"],
            default=["like", "comment"],
        )
        time_bin = st.selectbox("Time Bin", ["hour", "day", "week", "month"], index=1)

        df_action = get_action_data(conn, selected_actions, freq_map[time_bin])
        if df_action is not None:
            fig = plot_action_animation(df_action)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No action data found for selected actions.")

    elif chart_type == "📈 Follower Trend":
        st.markdown("#### Follower Growth Over Time")
        top_k = st.slider("Number of Top Users", 1, 20, 5)
        trend = get_follower_trend(conn, top_k=top_k)
        fig = plot_follower_growth(trend)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "🔥 Post Popularity Flow":
        st.markdown("#### Post Popularity Tree")
        post_id = st.number_input("Enter Root Post ID", min_value=0, value=1)
        G = build_post_graph(conn, post_id)
        fig = plot_post_popularity_flow(G)
        if fig:
            st.pyplot(fig)
        else:
            st.info("No reposts found for this post.")

    elif chart_type == "🌐 Repost Network":
        st.markdown("#### User Repost Network")
        path = build_repost_network(conn)
        components.html(open(path).read(), height=600)

    elif chart_type == "💬 Comment Sentiment Timeline":
        st.markdown("#### Sentiment of Comments Over Time")
        sentiment = get_sentiment_timeline(conn)
        fig = plot_sentiment_timeline(sentiment)
        st.plotly_chart(fig, use_container_width=True)
