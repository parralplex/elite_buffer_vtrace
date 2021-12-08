import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from model.utils import weights_init_xavier, num_flat_features, normalized_columns_initializer, weights_init


class ModelNetwork(nn.Module):
    def __init__(self, actions_count, flags):
        super(ModelNetwork, self).__init__()
        self.actions_count = actions_count
        self.flags = flags

        self.conv1 = nn.Conv2d(4, 64, stride=(3, 3), kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3))

        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, actions_count)
        self.fc3 = nn.Linear(512, 1)

        self.apply(weights_init)
        self.fc2.weight.data = normalized_columns_initializer(
            self.fc2.weight.data, 0.001)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(
            self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        time_dim = False
        t_dim, batch_dim = 1, 1
        if inputs.ndim == 5:
            t_dim, batch_dim, *_ = inputs.shape
            x = flatten(inputs, 0, 1)
            time_dim = True
        else:
            x = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        value = self.fc3(x)

        if time_dim:
            logits = logits.view(t_dim, batch_dim, self.actions_count)
            value = value.view(t_dim, batch_dim)

        return logits, value


class StateTransformationNetwork(nn.Module):
    def __init__(self, flags):
        super(StateTransformationNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 64, stride=(3, 3), kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3))

        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.feature = nn.Linear(512, flags.feature_out_layer_size)

        self.apply(weights_init)

        self.eval()

    def forward(self, inputs):
        time_dim = False
        t_dim, batch_dim = 1, 1
        if inputs.ndim == 5:
            t_dim, batch_dim, *_ = inputs.shape
            x = flatten(inputs, 0, 1)
            time_dim = True
        else:
            x = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.feature(x)
        if time_dim:
            x = x.view(t_dim, batch_dim, x.shape[1])
        return x
