import asyncio
import logging
import os
from pathlib import Path

import nest_asyncio
import pandas as pd
import streamlit as st
from oasis.social_platform.typing import ActionType
from stqdm import stqdm

# c.f. https://sehmi-conscious.medium.com/got-that-asyncio-feeling-f1a7c37cab8b
nest_asyncio.apply()

logger = logging.getLogger(__name__)


if os.environ.get("DEBUG", "false").lower() == "true":
    logging.basicConfig(level=logging.DEBUG)
    import importlib

    from sns import simulation
    from sns.utils import temporal_graph

    importlib.reload(simulation)
    importlib.reload(temporal_graph)


from sns.simulation import SimulationConfig, simulate_twitter  # noqa: E402

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

STATE2DEFAULT = {
    "simu_configs": None,
    "edited_base_agent_info_df": None,
    "edited_exp_agent_info_df": None,
}

# NOTE: Setting `key` also sets `st.session_state[key]`
if __name__ == "__main__":
    for state_key, default_value in STATE2DEFAULT.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value

    st.set_page_config(page_title="Social Network Simulator", layout="wide")
    st.title("Social Network Simulator")

    st.sidebar.header("Simulation Settings")

    st.sidebar.multiselect(
        "Available Actions",
        options=[action.value for action in ActionType],
        default=DEFAULT_AVAILABLE_ACTIONS,
        key="available_actions",
    )

    # The uploaded `base_agent_info_file` is immediately used to build the `base_agent_info_df`
    # without the need to put it in `st.session_state`
    base_agent_info_file = st.sidebar.file_uploader("Upload Base Agent Information", type="csv")
    just_read_df = pd.read_csv(base_agent_info_file, index_col=0) if base_agent_info_file is not None else None
    # logger.debug(f"{just_read_df=}")
    base_agent_info_df = just_read_df if just_read_df is not None else pd.DataFrame(columns=AGENT_INFO_FIELDS)
    assert set(base_agent_info_df.columns) == set(AGENT_INFO_FIELDS), (
        f"{set(base_agent_info_df.columns)=} != {set(AGENT_INFO_FIELDS)=}"
    )

    st.sidebar.write("Base Agent Information")
    # `data_editor(key="data_editor")` only saves the changes to `st.session_state["data_editor"]`
    st.session_state["edited_base_agent_info_df"] = st.sidebar.data_editor(
        base_agent_info_df, num_rows="dynamic", key="edited_base_agent_info_df_changes"
    )
    logger.debug(f"{st.session_state['edited_base_agent_info_df']=}")

    # Main
    st.write("Agent Information for Comparison (Refreshing if base changes)")
    st.session_state["edited_exp_agent_info_df"] = st.data_editor(
        st.session_state["edited_base_agent_info_df"], num_rows="dynamic", key="edited_exp_agent_info_df_changes"
    )

    base_check_col, exp_check_col = st.columns(2)
    with base_check_col:
        st.checkbox("Base", value=True, key="simulate_base")
    with exp_check_col:
        st.checkbox("Experiment", value=True, key="simulate_exp")

    # Prepare simulation configurations based on UI selections

    st.number_input("Run for Timesteps", min_value=1, value=3, step=1, key="num_timesteps")
    st.checkbox("Sequential", value=True, key="sequential")
    if st.button("Run Selected Simulations"):
        simu_db_home = Path("./data/simu_db")
        viz_home = Path("./data/visualization")

        simu_configs = []
        if st.session_state["simulate_base"]:
            simu_configs.append(
                SimulationConfig(
                    agent_info=st.session_state["edited_base_agent_info_df"],
                    available_actions=st.session_state["available_actions"],
                    db_path=simu_db_home / "base.db",
                    visualization_home=viz_home / "base",
                    num_timesteps=st.session_state["num_timesteps"],
                )
            )
        if st.session_state["simulate_exp"]:
            simu_configs.append(
                SimulationConfig(
                    agent_info=st.session_state["edited_exp_agent_info_df"],
                    available_actions=st.session_state["available_actions"],
                    db_path=simu_db_home / "exp.db",
                    visualization_home=viz_home / "exp",
                    num_timesteps=st.session_state["num_timesteps"],
                )
            )

        loop = asyncio.get_event_loop()
        try:
            if st.session_state["sequential"]:
                for config in stqdm(simu_configs, desc="Running Simulations"):
                    # logger.debug(f"Running simulation with {config=}")
                    loop.run_until_complete(simulate_twitter(config))
            else:
                with st.spinner("Running Simulations..."):
                    loop.run_until_complete(asyncio.gather(*[simulate_twitter(config) for config in simu_configs]))
        except Exception as e:
            st.error(f"Error running simulations: {e}")
            raise e
        st.success("Simulations completed successfully")
