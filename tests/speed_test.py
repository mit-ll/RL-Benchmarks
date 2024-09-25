import os

import ray
from ray.rllib.models import ModelCatalog
from rich import print

from src.environments import install_mujoco_xml
from src.models import CustomTorchModelMlp
from src.utils import custom_log_creator
from src.utils import ray_stats
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
]


def run_training(
    address: str,
    tmpdir: str,
    custom_model: str = "mlp",
    train_batch_per_cpu: int = 1000,
    num_sgd_iter: int = 0,
    epochs: int = 1,
    save_path: str = "save/speed/",
):
    """Run a training iteration using the custom network model provided."""
    context = ray.init(address=address, _temp_dir=tmpdir, ignore_reinit_error=True)
    ray_stats(context)

    # Register custom models
    ModelCatalog.register_custom_model("mlp", CustomTorchModelMlp)

    try:
        for env_id in env_ids:
            ppo_config = get_ppo_config(
                custom_model=custom_model,
                env_id=env_id,
                train_batch_strat="train_batch_per_cpu",
                train_batch_per_cpu=train_batch_per_cpu,
                num_sgd_iter=num_sgd_iter,
            )

            # Generate a trainer
            ppo_trainer = ppo_config.build(
                logger_creator=custom_log_creator(
                    custom_path=save_path,
                    custom_str=f"{custom_model}_{env_id}",
                )
            )

            # Train for iterations
            for _ in range(epochs):
                results = ppo_trainer.train()

            ppo_trainer.stop()

    except Exception as e:
        raise ValueError(e)
    finally:
        ray.shutdown()

    print("-" * 90)
    print("Results:")
    print(
        f"[bold red]num_env_steps_sampled_throughput_per_sec: {results['num_env_steps_sampled_throughput_per_sec']}"
    )
    print("-" * 90)


def test_custom_option(request):
    # Get the address and tempdir from user input
    address = request.config.getoption("--address")
    tmpdir = request.config.getoption("--tmpdir")
    python_path = os.environ["PYTHONPATH"]

    # Display input commands
    print(f"Address: {address}")
    print(f"Temp Dir: {tmpdir}")
    print(f"PYTHONPATH: {python_path}")

    # Run the scaling test
    run_training(address=address, tmpdir=tmpdir)
