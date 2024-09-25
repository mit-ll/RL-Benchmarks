"""Environment wrappers for gym.  These are used to add additional functionality into the standard gym environments.
"""
import logging
import random
from typing import Union

import gymnasium as gym
from distutils.dir_util import copy_tree
from distutils.sysconfig import get_python_lib
from gymnasium.spaces import Box
from gymnasium.wrappers.frame_stack import FrameStack

from src.utils import env_validation


def install_mujoco_xml(
    user_assets_path: str = "std",
    gym_assets_path: str = "/gymnasium/envs/mujoco/assets",
):
    """This is a way to insert user modified XML files into the Mujoco environments.  Gymnasium does not offer a method to insert XML modification to Mujoco before version `1.0.0` so this is a temporary fix."""

    # Specify the location of XML files
    std_mujoco_xml_dir = f"src/assets/{user_assets_path}"
    gym_mujoco_xml_dir = get_python_lib() + gym_assets_path

    # Replace the destination XML files with the user XML files
    copy_tree(std_mujoco_xml_dir, gym_mujoco_xml_dir)


class EnvFramestack(gym.Env):
    """This is a vectorized custom environment with framestacking.  Note that the formatting of the reset and step outputs need to be defined correctly or you will see an error that states the structures are incompatible."""

    def __init__(self, config, render_mode: Union[str, None] = None):
        self.debug = True
        self.env_id = config["env_id"]
        n_frames = config["n_frames"]

        self.custom_env = FrameStack(
            gym.make(
                self.env_id,
                render_mode=render_mode,
            ),
            n_frames,
        )
        self.action_space = self.custom_env.action_space
        self.observation_space = self.custom_env.observation_space

        # Currently setup for box observation space
        assert isinstance(self.custom_env.observation_space, Box)

        # Check observation type is of Box type
        if self.debug:
            logging.info("EnvFramestack logging enabled")

    def reset(self, seed=None, options=None):
        random.seed(seed)
        obs, info = self.custom_env.reset()

        # obs = np.array(obs)

        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.custom_env.step(action)

        # observation = np.array(observation)

        # Debugger
        if self.debug:
            env_validation(
                env_id=self.env_id,
                env=self.custom_env,
                obs=observation,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                action=action,
            )

        return (observation, reward, terminated, truncated, info)
