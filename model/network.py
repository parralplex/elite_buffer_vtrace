import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from model.utils import weights_init_xavier, num_flat_features, normalized_columns_initializer


class ModelNetwork(nn.Module):
    def __init__(self, actions_count):
        super(ModelNetwork, self).__init__()
        self.actions_count = actions_count

        self.conv1 = nn.Conv2d(3, 64, stride=(2, 2), kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3))

        self.fc1 = nn.Linear(128 * 9 * 9, 512)

        self.fc2 = nn.Linear(512, actions_count)
        self.fc4 = nn.Linear(512, 1)

        self.apply(weights_init_xavier)
        self.fc2.weight.data = normalized_columns_initializer(
            self.fc2.weight.data, 0.01)
        self.fc2.bias.data.fill_(0)
        self.fc4.weight.data = normalized_columns_initializer(
            self.fc4.weight.data, 1.0)
        self.fc4.bias.data.fill_(0)

        self.train()

    def forward(self, inputs, t_b_on_feature_vec=False, no_feature_vec=False):
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

        x_logits = F.relu(self.fc1(x))

        if not no_feature_vec:
            feature_vec = x_logits
            if t_b_on_feature_vec:
                feature_vec = feature_vec.view(t_dim, batch_dim, feature_vec.shape[1])

        logits = self.fc2(x_logits)
        value = self.fc4(x_logits)
        if time_dim:
            logits = logits.view(t_dim, batch_dim, self.actions_count)
            value = value.view(t_dim, batch_dim)
        if no_feature_vec:
            return logits, value
        else:
            return logits, value, feature_vec

    def get_flatten_layer_output_size(self):
        return self.fc1.out_features
