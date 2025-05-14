import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import nest_asyncio
import pandas as pd
import streamlit as st
from oasis.social_platform.typing import ActionType
from stqdm import stqdm

# c.f. https://sehmi-conscious.medium.com/got-that-asyncio-feeling-f1a7c37cab8b
nest_asyncio.apply()

logger = logging.getLogger(__name__)


def setup_debug_mode():
    logging.basicConfig(level=logging.DEBUG)
    import importlib

    from sns import simulation
    from sns.utils import temporal_graph

    importlib.reload(simulation)
    importlib.reload(temporal_graph)


def disable_debug_mode():
    logging.basicConfig(level=logging.INFO)


def toggle_debug_mode():
    if st.session_state["debug_mode"]:
        setup_debug_mode()
    else:
        disable_debug_mode()


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

STATE2DEFAULT: dict[str, Any] = {"simu_configs": None, "agent_info": {}}

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
    st.session_state["agent_info"]["Base"] = st.sidebar.data_editor(
        base_agent_info_df, num_rows="dynamic", key="edited_base_agent_info_df_changes"
    )
    logger.debug(f"{st.session_state['edited_base_agent_info_df']=}")

    # Main
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

    st.toggle(
        "Debug Mode",
        on_change=toggle_debug_mode,
        key="debug_mode",
        value=os.environ.get("DEBUG", "false").lower() == "true",
    )
