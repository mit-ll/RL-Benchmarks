"""Environment verification tests.
"""
import os

import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from ray.rllib.utils import check_env

from src.environments import EnvFramestack
from src.environments import install_mujoco_xml
from src.utils import env_validation

# Ensure that standard Mujoco XML files are installed
install_mujoco_xml()

# Only set this if you are on a server without visual rendering!
os.environ["SDL_VIDEODRIVER"] = "dummy"

# List of environments to test
env_ids = [
    # -------------- Classic --------------
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    # -------------- Box2d --------------
    # "BipedalWalker-v3",
    # "CarRacing-v2",
    # "LunarLander-v2", # broken at the moment
    # -------------- Mujoco --------------
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "HumanoidStandup-v4",
    "Humanoid-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    "Reacher-v4",
    "Swimmer-v4",
    "Pusher-v4",
    "Walker2d-v4",
    # -------------- Atari --------------
    # "Pong-ramDeterministic-v4",
    # "Breakout-ramDeterministic-v4",
]


def test_check_environments():
    """Check for some basic environments in the gym registry."""

    # List of all available environments
    avail_envs = gym.envs.registration.registry.keys()

    for env in env_ids:
        assert env in avail_envs


def test_making_environment():
    """Test creating an environment and running several actions using the agent."""

    for env_id in env_ids:
        # create the environment
        env = gym.make(env_id)  # add render_mode="human" if desired

        # Generate observation
        observation, info = env.reset()

        # Iterate a few times
        for _ in range(1000):
            # Randomly select an action
            action = env.action_space.sample()

            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Check for valid observation values
            env_validation(
                env_id,
                env,
                observation,
                float(reward),
                terminated,
                truncated,
                info,
                action,
            )

            # Reset when game terminates
            if terminated or truncated:
                observation, info = env.reset()

            # Render output to screen
            # env.render()

        env.close()


def test_framestacking():
    """Test for framestacking of environments.  Framestacking is necessary when historical information from environments is needed."""

    for env_id in env_ids:
        # create the environment
        env = gym.make(env_id)  # add render_mode="human" if desired
        env = FrameStack(env, 10)  # add framestacking

        # Generate observation
        observation, info = env.reset()

        # Iterate a few times
        for _ in range(1000):
            # Randomly select an action
            action = env.action_space.sample()

            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)

            # Check for valid observation values
            env_validation(
                env_id,
                env,
                observation,
                float(reward),
                terminated,
                truncated,
                info,
                action,
            )

            # Reset when game terminates
            if terminated or truncated:
                observation, info = env.reset()

            # Render output to screen
            # env.render()

        env.close()


def test_custom_environment():
    """Create a custom environment and verify that it can run."""

    # Create a customized environment
    env = EnvFramestack({"env_id": "CartPole-v1", "n_frames": 2})

    check_env(env)

    # Generate observation
    observation, info = env.reset()

    # Iterate a few times
    for _ in range(1000):
        # Randomly select an action
        action = env.action_space.sample()

        # Take step in environment
        observation, reward, terminated, done, info = env.step(action)

        # Check for valid observation values
        env_validation(
            env_id="CartpoleFramestack",
            env=env,
            obs=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=done,
            info=info,
            action=action,
        )

        # Reset when game terminates
        if terminated or done:
            observation, info = env.reset()

    env.close()
