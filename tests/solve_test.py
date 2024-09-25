import ray
from ray import air
from ray import tune
from ray.rllib.models import ModelCatalog

from src.environments import install_mujoco_xml
from src.models import CustomTorchModelMlp
from src.utils import ray_stats
from tests.utils_test import get_ppo_config

# Install Mujoco XML files
install_mujoco_xml()


def run_solver(
    address: str,
    tmpdir: str,
    custom_model: str,
    envs: dict,
    save_path: str = "save/solve/",
):
    """Runs the training algorithm until the environment is solved.  This is complicated by the fact that environments are randomized meaning the terminal reward can vary.

    In order to determine whether an algorithm has `solved` an environment we are using the state-of-the art benchmarks as defined by [**Tianshou**](https://github.com/ChenDRAG/mujoco-benchmark?tab=readme-ov-file).
    """

    context = ray.init(address=address, _temp_dir=tmpdir, ignore_reinit_error=True)
    ray_stats(context)

    ModelCatalog.register_custom_model("mlp", CustomTorchModelMlp)

    try:
        for env_id, reward_threshold in envs.items():
            ppo_config = get_ppo_config(
                custom_model=custom_model,
                env_id=env_id,
                train_batch_strat="train_batch_size",
                train_batch_size=1000,
                num_sgd_iter=10,
            )

            config = ppo_config

            tuner = tune.Tuner(
                "PPO",
                run_config=air.RunConfig(
                    storage_path=save_path,
                    name=f"{custom_model}_{env_id}",
                    stop={"episode_reward_mean": reward_threshold},
                ),
                param_space=config,
            )

            tuner.fit()

    except Exception as e:
        raise ValueError(e)
    finally:
        ray.shutdown()


def test_solve_cartpole(request):
    # Get the address and tempdir from user input
    address = request.config.getoption("--address")
    tmpdir = request.config.getoption("--tmpdir")

    # Environments with their corresponding reward thresholds
    envs = {"CartPole-v1": 400}

    run_solver(address, tmpdir, custom_model="mlp", envs=envs)
