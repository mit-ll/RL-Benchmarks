import torch
from ncps.torch import CfC as torch_CfC
from ncps.wirings import AutoNCP
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class CustomTorchModelMlp(TorchModelV2, torch.nn.Module):
    """Linear model for unit testing.

    Args:
        TorchModelV2 (_type_): _description_
        torch (_type_): _description_
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        torch.nn.Module.__init__(self)

        # Input space
        in_features = 1
        for dim in obs_space.shape:
            in_features = in_features * dim

        self.input = torch.nn.Linear(in_features=in_features, out_features=64)

        # Hidden layers
        self.linear1 = torch.nn.Linear(in_features=64, out_features=64)
        self.linear2 = torch.nn.Linear(in_features=64, out_features=64)

        # Activations
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

        # Heads
        self.action = torch.nn.Linear(in_features=64, out_features=num_outputs)
        self.value = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, input_dict, state, seq_lens):
        # Backbone
        x = self.input(input_dict["obs_flat"])
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)

        action_space = self.action(x)
        self.value_critic = self.value(x)

        return action_space, state

    def value_function(self):
        return torch.reshape(self.value_critic, (-1,))


class CustomTorchModelLstm(TorchRNN, torch.nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=64,
        lstm_state_size=64,
    ):
        torch.nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = torch.nn.Linear(self.obs_size, self.fc_size)
        self.lstm = torch.nn.LSTM(
            self.fc_size,
            self.lstm_state_size,
            batch_first=True,
        )
        self.relu1 = torch.nn.ReLU()
        self.action_branch = torch.nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = torch.nn.Linear(self.lstm_state_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # `h` and `c` need to be shape (I)
        h = self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        c = self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        hc = [h, c]

        return hc

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """
        `inputs` (B x T x I)
        `state` (B x I)

        We need to format `state` to (T x B x I) for processing.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """

        # Format inputs
        hx = torch.unsqueeze(state[0], 0)
        cx = torch.unsqueeze(state[1], 0)

        # Forward prop
        x = self.fc1(inputs)
        x = self.relu1(x)
        x, [h, c] = self.lstm(x, [hx, cx])

        # Format variables
        self._features = x
        action_out = self.action_branch(x)  # (B x T x I)
        h_out = torch.squeeze(h, 0)  # (B x I)
        c_out = torch.squeeze(c, 0)  # (B x I)

        return action_out, [h_out, c_out]


class CustomTorchModelCfc(TorchRNN, torch.nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        torch.nn.Module.__init__(self)

        self.cell_size = 64

        # Body
        wiring = AutoNCP(units=64, output_size=32)
        self.body = torch_CfC(obs_space.shape[0], wiring)

        # Activations
        self.relu1 = torch.nn.ReLU()

        # Heads
        self.action = torch.nn.Linear(in_features=32, out_features=num_outputs)
        self.value = torch.nn.Linear(in_features=32, out_features=1)

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Pass into body
        x, hx = self.body(input=inputs, hx=state[0])
        x = self.relu1(x)

        # Output heads
        actor_value = self.action(x)
        self.critic_value = self.value(x)

        return actor_value, [hx]

    @override(ModelV2)
    def get_initial_state(self):
        return [torch.zeros(self.cell_size)]

    def value_function(self):
        return torch.reshape(self.critic_value, (-1,))
