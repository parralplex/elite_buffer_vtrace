import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import flatten
from model.utils import weights_init_xavier, num_flat_features, normalized_columns_initializer, weights_init


class ModelNetwork(nn.Module):
    def __init__(self, actions_count, frames_stacked, feature_out_layer_size, use_additional_scaling_FC_layer=False):
        super(ModelNetwork, self).__init__()
        self.actions_count = actions_count
        self.use_additional_scaling_FC_layer = use_additional_scaling_FC_layer

        self.conv1 = nn.Conv2d(frames_stacked, 64, stride=(3, 3), kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3))

        if use_additional_scaling_FC_layer:
            self.fc1 = nn.Linear(128 * 6 * 6, 512)
            self.fc2 = nn.Linear(512, actions_count)
        else:
            self.fc1 = nn.Linear(128 * 6 * 6, feature_out_layer_size)
            self.fc2 = nn.Linear(feature_out_layer_size, actions_count)

        self.features = nn.Linear(512, feature_out_layer_size)

        self.fc3 = nn.Linear(512, 1)

        self.apply(weights_init)
        self.fc2.weight.data = normalized_columns_initializer(
            self.fc2.weight.data, 0.001)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(
            self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs, features: bool = False):
        time_dim = False
        t_dim, batch_dim = 1, 1
        if inputs.ndim == 5:
            t_dim, batch_dim, _, _, _ = inputs.shape
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

        feature_vec = torch.tensor([0])
        if features:
            if self.use_additional_scaling_FC_layer:
                feature_vec = self.features(x)
            else:
                feature_vec = x

        if time_dim:
            logits = logits.view(t_dim, batch_dim, self.actions_count)
            value = value.view(t_dim, batch_dim)
            if features:
                feature_vec = feature_vec.view(t_dim, batch_dim, feature_vec.shape[1])

        return logits, value, feature_vec
