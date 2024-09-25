import shutil

import numpy as np
import ray
from ray.rllib.models import ModelCatalog

from src.environments import install_mujoco_xml
from src.models import CustomTorchModelCfc
from src.models import CustomTorchModelLstm
from src.models import CustomTorchModelMlp
from src.utils import custom_log_creator
from src.utils import ray_stats
from src.utils import save_checkpoint
from tests.utils_test import get_ppo_config


# Install Mujoco XML files
install_mujoco_xml()

# Specify environments to use
env_ids = [
    # -------------- Classic --------------
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    # -------------- Mujoco --------------
    # benchmarks: https://github.com/ChenDRAG/mujoco-benchmark?tab=readme-ov-file
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "HumanoidStandup-v4",  # will produce NaNs if train_batch_size is too low
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


def run_training(
    address: str,
    tmpdir: str,
    custom_model: str,
    train_batch_size: int = 20000,
    num_sgd_iter: int = 20,
    epochs: int = 1,
    sgd_minibatch_size: int = 2048,
    save_path: str = "save/train",
):
    """Runs a training iteration across all the environments.  This is purely for testing purposes."""

    context = ray.init(address=address, _temp_dir=tmpdir, ignore_reinit_error=True)
    ray_stats(context)

    # Register custom models
    ModelCatalog.register_custom_model("mlp", CustomTorchModelMlp)
    ModelCatalog.register_custom_model("cfc", CustomTorchModelCfc)
    ModelCatalog.register_custom_model("lstm", CustomTorchModelLstm)

    try:
        for env_id in env_ids:
            ppo_config = get_ppo_config(
                custom_model=custom_model,
                env_id=env_id,
                train_batch_strat="train_batch_size",
                train_batch_size=train_batch_size,
                num_sgd_iter=num_sgd_iter,
                sgd_minibatch_size=sgd_minibatch_size,
            )

            # Generate a trainer
            ppo_trainer = ppo_config.build(
                logger_creator=custom_log_creator(
                    custom_path=save_path,
                    custom_str=f"{custom_model}_{env_id}",
                )
            )

            # Initialize
            checkpoint_dirs = []
            max_reward = -np.inf

            # Train for iterations
            for _ in range(epochs):
                result = ppo_trainer.train()

                # Check if reward is NaNs
                if np.isnan(result["episode_reward_mean"]):
                    raise ValueError("NaNs found in reward")

                # Saving checkpoints during training
                if result["episode_reward_mean"] > max_reward:
                    max_reward = result["episode_reward_mean"]
                    checkpoint_dir = save_checkpoint(result, ppo_trainer)
                    checkpoint_dirs.append(checkpoint_dir)

            # Delete checkpoints that are worse
            if len(checkpoint_dirs) > 1:
                checkpoint_dirs.pop()

                for delete_dir in checkpoint_dirs:
                    shutil.rmtree(delete_dir)

            ppo_trainer.evaluate()
            ppo_trainer.stop()

    except Exception as e:
        raise ValueError(e)
    finally:
        ray.shutdown()


def test_training_mlp(request):
    # Get the address and tempdir from user input
    address = request.config.getoption("--address")
    tmpdir = request.config.getoption("--tmpdir")

    run_training(address, tmpdir, custom_model="mlp", epochs=1)


def test_training_cfc(request):
    # Get the address and tempdir from user input
    address = request.config.getoption("--address")
    tmpdir = request.config.getoption("--tmpdir")

    run_training(address, tmpdir, custom_model="cfc", epochs=1)


def test_training_lstm(request):
    # Get the address and tempdir from user input
    address = request.config.getoption("--address")
    tmpdir = request.config.getoption("--tmpdir")

    run_training(address, tmpdir, custom_model="lstm", epochs=1)
