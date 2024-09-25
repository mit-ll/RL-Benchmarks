"""Common utils needed for repo."""
import logging
import os
import tempfile
from io import TextIOWrapper
from typing import Callable
from typing import Union

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.logger import UnifiedLogger


def ray_stats(context):
    """This function prints out relevant statistics that will help a developer
    debug what is going on within ray clusters.
    """

    tempdir = os.getenv("TMPDIR", "None")
    headnode = os.getenv("HEAD_NODE_ADDR", "None")

    # general info
    print("-" * 100)
    print("Ray Cluster Information".center(100))
    print("-" * 100)
    print(f"Dashboard IP: {context.dashboard_url}")
    print(f"Head node: {headnode}")
    print(f"TMPDIR: '{tempdir}'")
    print("Ray cluster resources = ", ray.cluster_resources())
    print("")

    # resources
    nodes = len(ray.nodes())
    cpus = ray.cluster_resources().get("CPU", 0)
    gpus = ray.cluster_resources().get("GPU", 0)
    obj_store_memory = ray.cluster_resources().get("object_store_memory", 0) / 1e9
    memory = ray.cluster_resources().get("memory", 0) / 1e9

    # logging
    print("Resources")
    print(f"Nodes:            {nodes:4}")
    print(f"CPUs:             {cpus:4.0f}")
    print(f"GPUs (Requested): {gpus:4.0f}       Note: Cluster may not have any GPUs...")
    print(f"Object Store Mem: {obj_store_memory:4.0f} GB")
    print(f"Memory:           {memory:4.0f} GB")


def save_checkpoint(result: dict, algorithm: Algorithm) -> str:
    """Saves a checkpoint for an Algorithm.

    Args:
        result (dict): A summary dict of the current algorithm.
        algorithm (Algorithm): The RLLib algorithm to checkpoint.

    Returns:
        str: The path to the checkpoint.
    """
    # Results
    print(f"episode_reward_mean: {result['episode_reward_mean']}")

    # Save model
    checkpoint_dir = algorithm.save(checkpoint_dir=f"{algorithm.logdir}")
    print(f"Checkpoint saved in directory {checkpoint_dir}")
    return checkpoint_dir


def env_validation(
    env_id: str,
    env: gym.Env,
    obs: np.ndarray,
    reward: Union[int, float],
    terminated: bool,
    truncated: bool,
    info: dict,
    action: np.ndarray,
):
    """Validation function for environments.

    Args:
        env_id (str): The environment Id.
        env (gym.Env): The environment object.
        obs (np.ndarray): Sample observation from environment.
        reward (Union[int, float]): Sample reward from environment.
        terminated (bool): Sample termination from environment.
        truncated (bool): Sample truncation from environment.
        info (dict): Sample info dict from environment.
        action (np.ndarray): Sample action from environment.

    Raises:
        ValueError: Raises error if any samples from environment are invalid.
    """
    check1 = np.any(np.isnan(obs))
    check2 = np.any(np.isinf(obs))
    check3 = np.any(np.isnan(action))
    check4 = np.any(np.isinf(action))
    check5 = np.any(np.isnan(reward))
    check6 = np.any(np.isinf(reward))
    check7 = np.any(np.isnan(terminated))
    check8 = np.any(np.isinf(terminated))
    check9 = np.isnan(truncated)
    check10 = np.isinf(truncated)
    check11 = not isinstance(reward, (int, float))
    check12 = not isinstance(terminated, bool)
    check13 = not isinstance(truncated, bool)
    check14 = not isinstance(info, dict)
    check15 = not env.observation_space.contains(obs)
    check16 = not env.action_space.contains(action)

    print_state = False

    if (
        check1
        or check2
        or check3
        or check4
        or check5
        or check6
        or check7
        or check8
        or check9
        or check10
    ):
        logging.error("NaN/Inf detected in the environment!")
        print_state = True

    if check11 | check12 | check13 | check14:
        logging.error("Invalid type detected in environment!")
        print_state = True

    if check15:
        logging.error(
            f"Obs not valid range | obs: {obs}, obs_range: {env.observation_space}"
        )

    if check16:
        logging.error(
            f"Action not valid range | action: {action}, obs_range: {env.action_space}"
        )

    if print_state:
        logging.error(f"environment: {env_id}")
        logging.error(f"obs: {obs}")
        logging.error(f"reward: {reward}")
        logging.error(f"terminated: {terminated}")
        logging.error(f"truncated: {truncated}")
        logging.error(f"info: {info}")
        logging.error(f"action: {action}")

    if (
        check1
        or check2
        or check3
        or check4
        or check5
        or check6
        or check7
        or check8
        or check9
        or check10
        or check11
        or check12
        or check13
        or check14
        or check15
        or check16
    ):
        raise ValueError("Invalid values detected in environment!")


def custom_log_creator(custom_path: str, custom_str: str) -> Callable:
    """Custom logger for RLLib.

    Args:
        custom_path (str): Path to custom log.
        custom_str (str): String of custom log.

    Returns:
        Callable: Reference to custom logger.
    """
    # Original time logging
    # timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    # logdir_prefix = "{}_{}".format(custom_str, timestr)

    # Custom string only
    logdir_prefix = "{}_".format(custom_str)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def log_algo_config(
    config: dict,
    file_ptr: TextIOWrapper,
    tabs: int = 0,
):
    """Designed to unpack the configuration into human readable attribtues.

    Args:
        config (AlgorithmConfig): An algorithms configuration file
        tabs (int, optional): _description_. Defaults to 0.
    """

    N_SPACING = 50
    N_INDENT = 4

    # Iterate over all items
    for key, val in config.items():
        # Found a nested dict
        if isinstance(val, dict):
            if len(val) == 0:
                pass  # no recursion into empty dict
            else:
                # Print out the key for the dict
                file_ptr.write(f"{' '*N_INDENT*tabs}{str(key).ljust(N_SPACING,' ')}\n")

                # Print out the key and vals for the dict
                log_algo_config(config=val, file_ptr=file_ptr, tabs=tabs + 1)
                continue

        # Print out the key and value for dict entry
        file_ptr.write(f"{' '*N_INDENT*tabs}{str(key).ljust(N_SPACING,'.')}| {val}\n")
