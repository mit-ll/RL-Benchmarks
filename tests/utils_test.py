import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig

from src.environments import EnvFramestack


def get_ppo_config(
    custom_model: str,
    env_id: str,
    train_batch_strat: str = "train_batch_per_cpu",  # "train_batch_per_cpu" or "train_batch_size"
    train_batch_per_cpu: int = 1000,
    train_batch_size: int = 20000,
    num_sgd_iter: int = 0,
    sgd_minibatch_size: int = 128,
):
    # Determine how many CPUs are on Ray Cluster
    n_cluster_cpus = int(ray.cluster_resources().get("CPU", 0))

    # Determine strategy.  "train_batch_per_cpu" is better for speed benchmarks whereas "train_batch_size" is better for training.
    match train_batch_strat:
        case "train_batch_size":
            pass
        case "train_batch_per_cpu":
            train_batch_size = train_batch_per_cpu * int(n_cluster_cpus * 0.95) - 2
        case _:
            raise ValueError("Invalid train batch strategy!")

    ppo_config = PPOConfig()

    # Configure environment
    if custom_model in ["mlp"]:
        ppo_config = ppo_config.environment(
            env=EnvFramestack,
            env_config={
                "env_id": env_id,
                "n_frames": 10,
            },
        )
    else:
        ppo_config = ppo_config.environment(env=env_id)

    ppo_config = ppo_config.training(
        grad_clip=40,
        train_batch_size=train_batch_size,
        sgd_minibatch_size=sgd_minibatch_size,
        num_sgd_iter=num_sgd_iter,
        model={"custom_model": custom_model, "max_seq_len": 10},
    )
    ppo_config = ppo_config.resources(
        num_gpus=torch.cuda.device_count(),
        placement_strategy="SPREAD",
    )
    ppo_config = ppo_config.rollouts(
        num_rollout_workers=int(0.95 * n_cluster_cpus) - 2,
        num_envs_per_worker=1,
        batch_mode="complete_episodes",
    )
    ppo_config = ppo_config.framework("torch")
    ppo_config = ppo_config.evaluation(
        evaluation_parallel_to_training=False,
        evaluation_interval=1,
        evaluation_num_workers=1,
    )
    # ppo_config = ppo_config.debugging(log_level="INFO")

    return ppo_config
