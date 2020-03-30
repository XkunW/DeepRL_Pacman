import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


class DQN_torch(nn.Module):

    def __init__(self, params):
        super(DQN_torch, self).__init__()
        self.name = 'q_network'
        self.params = params

        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(self.params['width']), kernel_size=2)
        conv_h = conv2d_size_out(conv2d_size_out(self.params['height']), kernel_size=2)

        self.flatten_input_size = conv_w * conv_h * 32
        self.fc1 = nn.Linear(self.flatten_input_size, 256)
        self.fc2 = nn.Linear(256, self.params['num_of_actions'])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.flatten_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_model_name(self):
        return "{}_{}".format(self.name, datetime.now())

    def save_ckpt(self):
        torch.save(self.state_dict(), self.get_model_name())

    def get_curr_performance(self):
        """
        Evaluate current Q net performance by running a game and get final score
        :return: Game score
        """
        pass

    def load_model(self, model_path):
        """
        Load a pre-trained model. Called if self.params['load_model'] is True
        :param model_path: Pre-trained model path
        """
        state = torch.load(model_path)
        self.load_state_dict(state)
