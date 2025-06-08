import asyncio
import logging
import os
import sqlite3
import uuid
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

    st.header("🔮 Simulation Settings: Compare Base vs. Experiment")
    st.markdown(
        """
        Upload the agent information for your **Base** simulation on the left.
        On the right, you can edit the **Experiment** configuration derived from the base.
        This allows you to easily compare and adjust settings between the two scenarios.
        """
    )

    with st.expander("📋 CSV Template & Format Guide", expanded=False):
        # Provide local file data/agent_info/False_Business_0.csv
        st.download_button(
            label="📥 Download Example CSV",
            data=open("data/agent_info/False_Business_0.csv", "rb").read(),
            file_name="False_Business_0.csv",
            mime="text/csv",
            help="Download this example CSV file to get started",
        )
        st.markdown("**Sample CSV Template:**")

        # Create a sample dataframe for demonstration
        sample_data = {
            "name": ["Alice", "Bob", "Charlie"],
            "previous_tweets": ['["Hello world!", "Good morning!"]', '["Nice day today"]', "[]"],
            "description": [
                "Tech enthusiast and coffee lover",
                "Sports fan and weekend hiker",
                "Artist and creative writer",
            ],
            "user_id": [0, 1, 2],
            "username": ["alice_tech", "bob_sports", "charlie_art"],
            "following_agentid_list": ["[1, 2]", "[0]", "[]"],
            "user_char": ["Optimistic and tech-savvy", "Outgoing and athletic", "Creative and introspective"],
        }

        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

        st.markdown("""
        **Important Notes:**
        - `previous_tweets` and `following_agentid_list` should be formatted as Python lists (use quotes and brackets)
        - `user_id` should be unique integers starting from 0
        - `num_followers` and `num_following` will be automatically calculated based on `following_agentid_list`
        """)

    base_agent_info_file = st.file_uploader("Upload Base CSV", type="csv")

    # Handle file loading with proper error handling
    if base_agent_info_file:
        try:
            base_agent_info_df = read_agent_info(base_agent_info_file)
            st.session_state["agent_info"]["Base"] = base_agent_info_df.copy()
            st.session_state["agent_info"]["Experiment"] = base_agent_info_df.copy()
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.markdown(
                """
            **Expected CSV format:**
            - Columns: {columns}
            - `previous_tweets`: List of strings, e.g., `["tweet1", "tweet2"]` or `[]`
            - `following_agentid_list`: List of integers, e.g., `[1, 2, 3]` or `[]`
            - `user_id`: Integer values
            """.format(columns=", ".join(AGENT_INFO_FIELDS))
            )
            base_agent_info_df = pd.DataFrame(columns=AGENT_INFO_FIELDS)
    else:
        # Provide a default set of agents for testing
        st.info(
            "💡 No CSV uploaded. Using default agent set for testing. You can edit these or upload your own CSV above."
        )
        default_data = {
            "name": ["Alice Cooper", "Bob Johnson", "Charlie Smith"],
            "num_followers": [0, 0, 0],  # Will be calculated
            "num_following": [0, 0, 0],  # Will be calculated
            "previous_tweets": [
                ["Hello everyone! Excited to be here.", "Having a great day!"],
                ["Welcome to my profile!", "Love connecting with new people."],
                ["Creative minds think alike!", "Art is everywhere."],
            ],
            "description": [
                "Tech enthusiast and coffee lover. Always learning something new!",
                "Sports fan and weekend hiker. Life is an adventure!",
                "Artist and creative writer. Finding beauty in everyday moments.",
            ],
            "user_id": [0, 1, 2],
            "username": ["alice_tech", "bob_sports", "charlie_art"],
            "following_agentid_list": [[1, 2], [0], []],
            "user_char": [
                "Optimistic, tech-savvy, and always eager to help others",
                "Outgoing, athletic, and loves outdoor activities",
                "Creative, introspective, and passionate about arts",
            ],
        }

        base_agent_info_df = pd.DataFrame(default_data)
        # Process the default data through the same pipeline
        base_agent_info_df["num_following"] = base_agent_info_df["following_agentid_list"].apply(len)
        base_agent_info_df["num_followers"] = 0
        for idx, row in base_agent_info_df.iterrows():
            following_ids = row["following_agentid_list"]
            if following_ids:
                for following_id in following_ids:
                    if following_id in base_agent_info_df.index:
                        base_agent_info_df.at[following_id, "num_followers"] += 1

    col1, col2 = st.columns(2)

    with col1:
        # Handle base agent information with validation
        st.markdown("### 📥 Base")
        if not set(base_agent_info_df.columns) == set(AGENT_INFO_FIELDS):
            st.error("The uploaded CSV file does not contain the required columns.")
        else:
            # Initialize base dataframe in session state if not already present
            if "Base" not in st.session_state["agent_info"]:
                st.session_state["agent_info"]["Base"] = base_agent_info_df.copy()

            # Use the current session state data for the data_editor
            current_base_data = st.session_state["agent_info"]["Base"]

            # Render data editor with current session state data
            edited_df = st.data_editor(
                current_base_data,
                num_rows="dynamic",
                key="edited_base_agent_info_df_changes",
                use_container_width=True,
                hide_index=False,
            )

            # Store the edited dataframe in session state
            st.session_state["agent_info"]["Base"] = edited_df

            user_sel_col, info_col = st.columns(2)
            with user_sel_col:
                posting_user_name = st.selectbox(
                    "Select User to Post",
                    options=st.session_state["agent_info"]["Base"]["name"].tolist()
                    if not st.session_state["agent_info"]["Base"].empty
                    else [],
                    key="posting_user_base",
                    help="Choose which user will create the post",
                )

            # Show user information and posting interface after user selection
            if posting_user_name:
                # Get user information from session state dataframe
                user_row = st.session_state["agent_info"]["Base"][
                    st.session_state["agent_info"]["Base"]["name"] == posting_user_name
                ]
                with info_col:
                    if not user_row.empty:
                        user_info = user_row.iloc[0]

                    num_followers = int(user_info["num_followers"]) if pd.notna(user_info["num_followers"]) else 0
                    num_following = int(user_info["num_following"]) if pd.notna(user_info["num_following"]) else 0
                    st.markdown(f"Followers: *{num_followers}*")
                    st.markdown(f"Following: *{num_following}*")
                with st.expander("##### ℹ️ Profile", expanded=True):
                    st.text(user_info["description"])

                # Show previous tweets if they exist (from session state)
                current_user_data = st.session_state["agent_info"]["Base"][
                    st.session_state["agent_info"]["Base"]["name"] == posting_user_name
                ]
                if not current_user_data.empty:
                    current_tweets = current_user_data.iloc[0]["previous_tweets"]
                    if isinstance(current_tweets, list) and len(current_tweets) > 0:
                        with st.expander("##### 📋 Previous Posts", expanded=True):
                            for tweet in reversed(current_tweets):
                                st.info(tweet)
                    else:
                        st.text("##### 📭 No previous posts")

                st.divider()

                post_content = st.text_area(
                    label="Post Content",
                    placeholder="Input your base test case here...",
                    key="base_test_prompt",
                    help="Enter the content for the post",
                )

                if st.button("Post", type="secondary", key="post_button_base", icon="💬", use_container_width=True):
                    # Find the user's index in the session state dataframe
                    user_index = st.session_state["agent_info"]["Base"][
                        st.session_state["agent_info"]["Base"]["name"] == posting_user_name
                    ].index
                    if len(user_index) > 0:
                        idx = user_index[0]
                        # Apply the post content to the selected user's previous_tweets
                        st.session_state["agent_info"]["Base"].at[idx, "previous_tweets"].append(post_content)
                        st.success(f"✅ Post created for {posting_user_name}!")
                        # Force a rerun to refresh the data_editor
                        st.rerun()
                    else:
                        st.error(f"❌ User {posting_user_name} not found in the table")

    with col2:
        # Define prerequisite check function for experiment
        def check_base_prerequisites():
            return "agent_info" in st.session_state and "Base" in st.session_state["agent_info"]

        # Inlined render_agent_info_editor for Experiment scenario
        st.markdown("### 🧪 Experiment")

        # Check prerequisites
        if not check_base_prerequisites():
            st.info("Please upload and edit the Base Agent Information first.")
            st.session_state["agent_info"]["Experiment"] = pd.DataFrame(columns=AGENT_INFO_FIELDS)
        else:
            # Initialize experiment dataframe from Base if not already in session state
            if "Experiment" not in st.session_state["agent_info"]:
                st.session_state["agent_info"]["Experiment"] = st.session_state["agent_info"]["Base"].copy()

            # Use the current session state data for the data_editor
            current_experiment_data = st.session_state["agent_info"]["Experiment"]

            # Render data editor with current session state data
            edited_df = st.data_editor(
                current_experiment_data,
                num_rows="dynamic",
                key="edited_exp_agent_info_df_changes",
                use_container_width=True,
                hide_index=False,
            )

            # Store the edited dataframe in session state
            st.session_state["agent_info"]["Experiment"] = edited_df

            user_sel_col, info_col = st.columns(2)
            with user_sel_col:
                posting_user_name = st.selectbox(
                    "Select User to Post",
                    options=current_experiment_data["name"].tolist() if not current_experiment_data.empty else [],
                    key="posting_user_experiment",
                    help="Choose which user will create the post",
                )

            # Show user information and posting interface after user selection
            if posting_user_name:
                # Get user information from current session state data
                user_row = current_experiment_data[current_experiment_data["name"] == posting_user_name]
                with info_col:
                    if not user_row.empty:
                        user_info = user_row.iloc[0]

                    num_followers = int(user_info["num_followers"]) if pd.notna(user_info["num_followers"]) else 0
                    num_following = int(user_info["num_following"]) if pd.notna(user_info["num_following"]) else 0
                    st.markdown(f"Followers: *{num_followers}*")
                    st.markdown(f"Following: *{num_following}*")
                with st.expander("##### ℹ️ Profile", expanded=True):
                    st.text(user_info["description"])

                # Show previous tweets if they exist (from session state)
                current_user_data = st.session_state["agent_info"]["Experiment"][
                    st.session_state["agent_info"]["Experiment"]["name"] == posting_user_name
                ]
                if not current_user_data.empty:
                    current_tweets = current_user_data.iloc[0]["previous_tweets"]
                    if isinstance(current_tweets, list) and len(current_tweets) > 0:
                        with st.expander("##### 📋 Previous Posts", expanded=True):
                            for tweet in reversed(current_tweets):
                                st.info(tweet)
                    else:
                        st.text("##### 📭 No previous posts")

                st.divider()

                post_content = st.text_area(
                    label="Post Content",
                    placeholder="Input your experimental test case here...",
                    key="experiment_test_prompt",
                    help="Enter the content for the post",
                )

                if st.button(
                    "Post", type="secondary", key="post_button_experiment", icon="💬", use_container_width=True
                ):
                    # Find the user's index in the session state dataframe
                    user_index = st.session_state["agent_info"]["Experiment"][
                        st.session_state["agent_info"]["Experiment"]["name"] == posting_user_name
                    ].index
                    if len(user_index) > 0:
                        idx = user_index[0]
                        # Apply the post content to the selected user's previous_tweets
                        st.session_state["agent_info"]["Experiment"].at[idx, "previous_tweets"].append(post_content)
                        print(st.session_state["agent_info"]["Experiment"].at[idx, "previous_tweets"])
                        st.success(f"✅ Post created for {posting_user_name}!")
                        # Force a rerun to refresh the data_editor
                        st.rerun()
                    else:
                        st.error(f"❌ User {posting_user_name} not found in the table")

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

    freq_map = {"hour": "H", "day": "D", "week": "W", "month": "M"}

    simu_id_to_analyze = st.selectbox(
        "Select Simulation to Analyze",
        options=["Base", "Experiment"],
        index=0,
        key="simu_id_to_analyze",
    )

    # Use the current run's UUID if available, otherwise look for any matching files
    if "current_run_uuid" in st.session_state:
        run_uuid = st.session_state["current_run_uuid"]
        db_path = Path(f"./data/simu_db/{simu_id_to_analyze}_{run_uuid}.db")
    else:
        # Fallback: look for any database file matching the simulation ID pattern
        simu_db_home = Path("./data/simu_db")
        pattern_files = list(simu_db_home.glob(f"{simu_id_to_analyze}_*.db"))
        if pattern_files:
            # Use the most recent file
            db_path = max(pattern_files, key=lambda p: p.stat().st_mtime)
            st.info(f"Using database: {db_path.name}")
        else:
            db_path = Path(f"./data/simu_db/{simu_id_to_analyze}.db")  # Original fallback

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
