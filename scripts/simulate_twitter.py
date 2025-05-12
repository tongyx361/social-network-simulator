import asyncio
import os

import oasis
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis import ActionType, EnvAction, SingleAction


async def main():
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.ZHIPU,
        model_type=ModelType.GLM_4_FLASH,
    )

    # Define the available actions for the agents
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
    ]

    # Define the path to the database
    db_path = "./data/twitter_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        agent_profile_path=("data/twitter/anonymous_topic_200_1h/False_Business_0.csv"),
        agent_models=openai_model,
        available_actions=available_actions,
    )

    # Run the environment
    await env.reset()

    action_1 = SingleAction(agent_id=0, action=ActionType.CREATE_POST, args={"content": "Earth is flat."})
    env_actions_1 = EnvAction(
        # Activate 5 agents with id 1, 3, 5, 7, 9
        activate_agents=[1, 3, 5, 7, 9],
        intervention=[action_1],
    )

    action_2 = SingleAction(
        agent_id=1,
        action=ActionType.CREATE_POST,
        args={"content": "Earth is not flat."},
    )
    env_actions_2 = EnvAction(activate_agents=[2, 4, 6, 8, 10], intervention=[action_2])

    empty_action = EnvAction()  # Means activate all agents and no intervention

    all_env_actions = [
        env_actions_1,
        env_actions_2,
        empty_action,
    ]

    # Simulate 3 timesteps
    for i in range(3):
        env_actions = all_env_actions[i]
        # Perform the actions
        await env.step(env_actions)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
