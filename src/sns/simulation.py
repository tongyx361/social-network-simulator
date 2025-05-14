import asyncio
import logging
import os
from pathlib import Path

import pandas as pd
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from colorama import Back
from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

logger = logging.getLogger(__name__)


async def simulate_twitter(
    available_actions: list[ActionType],
    agent_info: pd.DataFrame,
    visualization_home: Path | None = None,
    db_path: str = ":memory:",
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    start_time = 0
    clock = Clock(k=clock_factor)
    twitter_channel = Channel()
    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )
    twitter_task = asyncio.create_task(infra.running())
    models = [
        ModelFactory.create(
            model_platform=ModelPlatformType.ZHIPU,
            model_type=ModelType.GLM_4_FLASH,
        )
    ]

    agent_graph = await generate_agents(
        agent_info=agent_info,
        twitter_channel=twitter_channel,
        start_time=start_time,
        model=models,
        recsys_type=recsys_type,
        available_actions=available_actions,
        twitter=infra,
    )
    if visualization_home is not None:
        agent_graph.visualize(visualization_home / "initial_social_graph.png")

    for timestep in range(1, num_timesteps + 1):
        clock.time_step = timestep * 3
        logger.info(f"timestep:{timestep}")
        db_file = db_path.split("/")[-1]
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        # if you want to disable recsys, please comment this line
        await infra.update_rec_table()

        tasks = []
        for node_id, agent in agent_graph.get_agents():
            tasks.append(agent.perform_action_by_llm())

        await asyncio.gather(*tasks)
        if visualization_home is not None:
            agent_graph.visualize(visualization_home / f"timestep_{timestep}_social_graph.png")

    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await twitter_task
