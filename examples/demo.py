import logging
import os

import pandas as pd
import streamlit as st
from oasis.social_platform.typing import ActionType

logger = logging.getLogger(__name__)


if os.environ.get("DEBUG", "false").lower() == "true":
    logger.setLevel(logging.DEBUG)
    import importlib

    from sns import simulation
    from sns.utils import temporal_graph

    importlib.reload(simulation)
    importlib.reload(temporal_graph)


AGENT_INFO_FIELDS = [
    "user_id",
    "name",
    "username",
    "following_agentid_list",
    "previous_tweets",
    "user_char",
    "description",
]

if __name__ == "__main__":
    st.set_page_config(page_title="Social Network Simulator", layout="wide")
    st.title("Social Network Simulator")

    st.sidebar.header("Simulation Settings")
    # Use session state to keep track of num_timesteps if needed across runs/interactions
    if "num_timesteps" not in st.session_state:
        st.session_state.num_timesteps = 3  # Default value

    # Update session state when number input changes
    st.session_state.num_timesteps = st.sidebar.number_input(
        "Number of Timesteps", min_value=1, value=st.session_state.num_timesteps, step=1, key="num_timesteps_input"
    )
    num_timesteps = st.session_state.num_timesteps  # Get value for local use

    available_actions = st.sidebar.multiselect(
        "Available Actions",
        options=[action.value for action in ActionType],
        default=[
            ActionType.DO_NOTHING.value,
            ActionType.REPOST.value,
            ActionType.LIKE_POST.value,
            ActionType.FOLLOW.value,
        ],
        key="available_actions_select",
    )

    base_agent_info_file = st.sidebar.file_uploader("Upload Base Agent Information", type="csv")
    base_agent_info_df = (
        pd.read_csv(base_agent_info_file, index_col=0)
        if base_agent_info_file is not None
        else pd.DataFrame(columns=AGENT_INFO_FIELDS)
    )
    logger.debug(f"{base_agent_info_df=}")
    assert set(base_agent_info_df.columns) == set(AGENT_INFO_FIELDS)
    st.sidebar.write("Base Agent Information:")
    edited_base_agent_info_df = st.sidebar.data_editor(base_agent_info_df, key="base_agent_info", num_rows="dynamic")

    st.write("Agent Information for Comparison:")
    edited_agent_info = st.data_editor(
        edited_base_agent_info_df,  # Following the change of edited_base_agent_info_df
        key="edited_agent_info",
        num_rows="dynamic",
    )
