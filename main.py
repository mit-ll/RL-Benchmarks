import datetime
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional
from typing import Type

import numpy as np
import ray
import torch
import typer
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from typing_extensions import Annotated

from src.environments import EnvFramestack
from src.environments import install_mujoco_xml
from src.models import CustomTorchModelCfc
from src.models import CustomTorchModelLstm
from src.models import CustomTorchModelMlp
from src.utils import custom_log_creator
from src.utils import log_algo_config
from src.utils import ray_stats
from src.utils import save_checkpoint

# Install Mujoco XML files
install_mujoco_xml()

# Config
LOGGING = True

# Write logs
if LOGGING:
    log_path = "log"
    current_time = str(datetime.datetime.now()).replace(" ", "_")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(
        filename=f"log/{current_time}.log", filemode="w", level=logging.INFO
    )


@dataclass
class Config:
    network_id: str
    custom_network: Type
    epochs: int = 1
    train_batch_size: int = 100000
    sgd_minibatch_size: int = 2048
    num_sgd_iter: int = 25
    seq_len: int = 10
    num_envs_per_worker: int = 4
    eval_workers: int = 1
    framework: str = "torch"
    save_checkpoints: bool = True
    seeds: int = 1
    grad_clip: int = 20  # default is 40, which can cause nans


# Configurable parameters
USER_CONFIGS = [
    Config(
        network_id="mlp",
        custom_network=CustomTorchModelMlp,
        epochs=50,
    ),
    Config(
        network_id="cfc",
        custom_network=CustomTorchModelCfc,
        epochs=50,
    ),
    Config(
        network_id="lstm",
        custom_network=CustomTorchModelLstm,
        epochs=50,
    ),
]


def get_env_ids() -> list:
    # Environments: https://gymnasium.farama.org/environments/classic_control/
    return [
        # -------------- Classic --------------
        "CartPole-v1",
        "Pendulum-v1",
        "Acrobot-v1",
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


def get_config(env_id: str, user_config: Config) -> AlgorithmConfig:
    # Determine how many CPUs are on Ray Cluster
    n_cluster_cpus = int(ray.cluster_resources().get("CPU", 0))

    # Create a training configuration
    ppo_config = PPOConfig()

    # Configure environment
    if user_config.network_id in ["mlp"]:
        ppo_config = ppo_config.environment(
            env=EnvFramestack,
            env_config={
                "env_id": env_id,
                "n_frames": user_config.seq_len,
            },
        )
    else:
        ppo_config = ppo_config.environment(env=env_id)

    ppo_config = ppo_config.resources(
        placement_strategy="SPREAD",
    )
    ppo_config = ppo_config.training(
        grad_clip=user_config.grad_clip,
        train_batch_size=user_config.train_batch_size,
        sgd_minibatch_size=user_config.sgd_minibatch_size,
        num_sgd_iter=user_config.num_sgd_iter,
        model={
            "custom_model": user_config.network_id,
            "max_seq_len": user_config.seq_len,
        },
    )
    ppo_config = ppo_config.rollouts(
        num_rollout_workers=int(0.95 * n_cluster_cpus),
        num_envs_per_worker=user_config.num_envs_per_worker,
        batch_mode="complete_episodes",
    )
    ppo_config = ppo_config.framework(user_config.framework)
    ppo_config = ppo_config.evaluation(
        evaluation_parallel_to_training=False,
        evaluation_interval=1,
        evaluation_num_workers=user_config.eval_workers,
    )
    ppo_config = ppo_config.debugging(log_level="WARN")

    # Try to recreate workers when they fail
    ppo_config = ppo_config.fault_tolerance(recreate_failed_workers=True)

    return ppo_config


def run_train_eval(env_id: str, algorithm: Algorithm, algo_config, user_config: Config):
    # Initialize
    checkpoint_dirs = []
    max_reward = -np.inf

    # Train for N iterations
    for ii in range(user_config.epochs):
        # Use a try catch for failed training attempts
        try:
            result = algorithm.train()
        except Exception as e:
            print(f"Training failed on: {env_id}")
            print(e)
            break

        # Saving checkpoints during training
        if user_config.save_checkpoints:
            if result["episode_reward_mean"] > max_reward:
                max_reward = result["episode_reward_mean"]
                checkpoint_dir = save_checkpoint(result, algorithm)
                checkpoint_dirs.append(checkpoint_dir)

    # Delete checkpoints that are worse
    if len(checkpoint_dirs) > 1:
        checkpoint_dirs.pop()

        for delete_dir in checkpoint_dirs:
            shutil.rmtree(delete_dir)

    # algo.evaluate()
    algorithm.stop()


def main(
    address: Annotated[Optional[str], typer.Option()] = None,
    tmpdir: Annotated[Optional[str], typer.Option()] = None,
):
    # ray start
    context = ray.init(address=address, _temp_dir=tmpdir)
    ray_stats(context)

    print("-" * 90)
    print("Python Script".center(90))
    print("-" * 90)

    # Register the environments to use
    env_ids = get_env_ids()

    try:
        for user_config in USER_CONFIGS:
            # Register the model to use
            ModelCatalog.register_custom_model(
                user_config.network_id,
                user_config.custom_network,
            )

            for seeds in range(user_config.seeds):
                for env_id in env_ids:
                    algo_config = get_config(env_id, user_config)

                    # Generate a trainer
                    algo = algo_config.build(
                        logger_creator=custom_log_creator(
                            custom_path="save/",
                            custom_str=f"{user_config.network_id}_{env_id}",
                        )
                    )

                    # Run the training
                    run_train_eval(
                        env_id=env_id,
                        algorithm=algo,
                        algo_config=algo_config,
                        user_config=user_config,
                    )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    typer.run(main)
