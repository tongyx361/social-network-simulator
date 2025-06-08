import ast
import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from colorama import Back
from tqdm import tqdm

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType, RecsysType

logger = logging.getLogger(__name__)


AGENT_INFO_FIELDS = [
    "name",
    "num_followers",
    "num_following",
    "previous_tweets",
    "description",
    "user_id",
    "username",
    "following_agentid_list",
    "user_char",
]


def read_agent_info(agent_info_path: Path | IO) -> pd.DataFrame:
    """Read and process agent information from CSV file.

    Args:
        agent_info_path: Path to CSV file or file-like object

    Returns:
        Processed DataFrame with agent information

    Raises:
        ValueError: If required columns are missing or data format is invalid
    """
    try:
        agent_info_df = pd.read_csv(agent_info_path, index_col=0)
    except (ValueError, IndexError):
        # Reset file pointer if it's a file-like object
        if hasattr(agent_info_path, "seek"):
            agent_info_path.seek(0)
        agent_info_df = pd.read_csv(agent_info_path)

    # Process user_id column - convert to int, handling NaN values
    if "user_id" in agent_info_df.columns:
        agent_info_df["user_id"] = pd.to_numeric(agent_info_df["user_id"], errors="coerce").fillna(0).astype(int)

    # Process previous_tweets column - safely parse list strings
    if "previous_tweets" in agent_info_df.columns:

        def safe_parse_tweets(value):
            if pd.isna(value) or value == "":
                return []
            if isinstance(value, list):
                return value
            try:
                parsed = ast.literal_eval(str(value))
                return parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                # If parsing fails, treat as a single tweet
                return [str(value)] if str(value).strip() else []

        agent_info_df["previous_tweets"] = agent_info_df["previous_tweets"].apply(safe_parse_tweets)

    # Process following_agentid_list column - safely parse list of integers
    if "following_agentid_list" in agent_info_df.columns:

        def safe_parse_following(value):
            if pd.isna(value) or value == "":
                return []
            if isinstance(value, list):
                return [int(x) for x in value if str(x).strip().isdigit()]
            try:
                parsed = ast.literal_eval(str(value))
                if isinstance(parsed, list):
                    return [int(x) for x in parsed if str(x).strip().isdigit()]
                elif str(parsed).strip().isdigit():
                    return [int(parsed)]
                else:
                    return []
            except (ValueError, SyntaxError):
                return []

        agent_info_df["following_agentid_list"] = agent_info_df["following_agentid_list"].apply(safe_parse_following)

    # Calculate num_following from following_agentid_list
    agent_info_df["num_following"] = agent_info_df["following_agentid_list"].apply(len)

    # Initialize and calculate num_followers
    agent_info_df["num_followers"] = 0
    for idx, row in agent_info_df.iterrows():
        following_ids = row["following_agentid_list"]
        if following_ids:
            for following_id in following_ids:
                # Make sure the following_id exists in the dataframe
                if following_id in agent_info_df.index:
                    agent_info_df.at[following_id, "num_followers"] += 1

    # Sort by followers (descending)
    agent_info_df = agent_info_df.sort_values(by="num_followers", ascending=False)

    # Check if all required columns are present
    missing_columns = set(AGENT_INFO_FIELDS) - set(agent_info_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Ensure all required columns are present and in the right order
    agent_info_df = agent_info_df[AGENT_INFO_FIELDS]

    return agent_info_df


@dataclass
class ModelConfig:
    model_platform: ModelPlatformType = ModelPlatformType.ZHIPU
    model_type: ModelType = ModelType.GLM_4_FLASH


@dataclass
class SimulationConfig:
    agent_info: pd.DataFrame
    db_path: Path
    model_config: ModelConfig = field(default_factory=ModelConfig)
    available_actions: list[ActionType | str] = field(
        default_factory=lambda: [
            ActionType.DO_NOTHING.value,
            ActionType.REPOST.value,
            ActionType.LIKE_POST.value,
            ActionType.FOLLOW.value,
        ]
    )
    visualization_home: Path | None = None
    num_timesteps: int = 3
    clock_factor: int = 60
    recsys_type: RecsysType = RecsysType.TWHIN
    pbar: tqdm | None = None


async def simulate_twitter(config: SimulationConfig) -> None:
    # print(f"Simulating Twitter with {config=}")
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
    Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)
    if config.visualization_home is not None:
        Path(config.visualization_home).mkdir(parents=True, exist_ok=True)

    start_time = 0
    clock = Clock(k=config.clock_factor)
    channel = Channel()
    infra = Platform(
        db_path=config.db_path,
        channel=channel,
        sandbox_clock=clock,
        recsys_type=config.recsys_type,
        start_time=start_time,
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )
    twitter_task = asyncio.create_task(infra.running())
    model_config = config.model_config
    models = [
        ModelFactory.create(
            model_platform=model_config.model_platform,
            model_type=model_config.model_type,
        )
    ]

    # print(f"Generating agent graph with {config.agent_info=}")
    agent_graph = await generate_agents(
        agent_info=config.agent_info,
        channel=channel,
        start_time=start_time,
        model=models,
        recsys_type=config.recsys_type,
        available_actions=config.available_actions,
        twitter=infra,
    )
    if config.visualization_home is not None:
        agent_graph.visualize(config.visualization_home / "initial_social_graph.png")

    # print(f"Running simulation for {config.num_timesteps=}")
    for timestep in range(1, config.num_timesteps + 1):
        clock.time_step = timestep * 3
        logger.info(f"timestep:{timestep}")
        db_file = config.db_path.name
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        # if you want to disable recsys, please comment this line
        await infra.update_rec_table()

        tasks = []
        for node_id, agent in agent_graph.get_agents():
            tasks.append(agent.perform_action_by_llm())

        await asyncio.gather(*tasks)
        if config.visualization_home is not None:
            agent_graph.visualize(config.visualization_home / f"timestep_{timestep}_social_graph.png")
        if config.pbar is not None:
            config.pbar.update(1)

    # print("Waiting for simulation to finish...")
    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await twitter_task


if __name__ == "__main__":
    asyncio.run(
        simulate_twitter(
            SimulationConfig(
                agent_info=pd.read_csv("./data/agent_info/False_Business_0.csv"),
                db_path=Path("./data/simu_db/test.db"),
                visualization_home=Path("./data/visualization/test"),
            )
        )
    )
