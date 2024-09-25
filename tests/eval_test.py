import gymnasium as gym
import numpy as np
import ray
from ray.rllib.models import ModelCatalog

from src.environments import EnvFramestack
from src.models import CustomTorchModelCfc
from src.models import CustomTorchModelLstm
from src.models import CustomTorchModelMlp
from tests.train_test import get_ppo_config


class TrainerClass:
    def __init__(self, framestacking: bool = False):
        ray.shutdown()
        ray.init()

        self.save_dir = "save"

        if framestacking:
            self.env = EnvFramestack(
                config={
                    "env_id": "CartPole-v1",
                    "n_frames": 10,
                },
                render_mode="human",
            )
        else:
            self.env = gym.make("CartPole-v1", render_mode="human")

    def train(self, stop_criteria):
        pass

    def load(self, path: str, custom_model: str, env_id: str):
        # Register custom models
        ModelCatalog.register_custom_model("mlp", CustomTorchModelMlp)
        ModelCatalog.register_custom_model("cfc", CustomTorchModelCfc)
        ModelCatalog.register_custom_model("lstm", CustomTorchModelLstm)

        ppo_config = get_ppo_config(custom_model=custom_model, env_id=env_id)

        ppo_trainer = ppo_config.build()
        ppo_trainer.restore(path)

        self.agent = ppo_trainer

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        env = self.env

        # run until episode ends
        episode_reward = 0
        terminated = False
        obs, info = env.reset()

        # range(2) b/c h- and c-states of the LSTM.
        previous_state = [np.zeros([64], np.float32) for _ in range(2)]
        previous_action = 0
        previous_reward = 0

        while not terminated:
            action, state_out, _ = self.agent.compute_single_action(
                observation=obs,
                state=previous_state,
                previous_action=previous_action,
                previous_reward=previous_reward,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Save previous state/action/reward
            previous_state = state_out
            previous_action = action
            previous_reward = reward

        return episode_reward


def test_trainer_mlp():
    # Create a trainer
    trainer = TrainerClass(framestacking=True)

    # Load saved
    checkpoint_path = "tests/assets/cartpole_checkpoint_mlp"
    trainer.load(path=checkpoint_path, custom_model="mlp", env_id="CartPole-v1")

    # Test loaded
    trainer.test()


def test_trainer_cfc():
    # Create a trainer
    trainer = TrainerClass(framestacking=False)

    # Load saved
    checkpoint_path = "tests/assets/cartpole_checkpoint_cfc"
    trainer.load(path=checkpoint_path, custom_model="cfc", env_id="CartPole-v1")

    # Test loaded
    trainer.test()


def test_trainer_lstm():
    # Create a trainer
    trainer = TrainerClass(framestacking=False)

    # Load saved
    checkpoint_path = "tests/assets/cartpole_checkpoint_lstm"
    trainer.load(path=checkpoint_path, custom_model="lstm", env_id="CartPole-v1")

    # Test loaded
    trainer.test()
