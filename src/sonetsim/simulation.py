import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

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
