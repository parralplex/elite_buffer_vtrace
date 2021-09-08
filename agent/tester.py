import gym
import torch
import torch.nn.functional as F
from model.network import ModelNetwork


class UnSupportedFormatError(Exception):
    pass


class Tester(object):
    def __init__(self, test_ep_count, model_save_uri, flags):
        self.flags = flags
        self.test_ep_count = test_ep_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Tester is using CUDA")
        else:
            print("CUDA IS NOT AVAILABLE")
        self.env = gym.make(self.flags.env)
        self.model = ModelNetwork(self.env.action_space.n).to(self.device).eval()
        loaded_dict = torch.load(model_save_uri)
        if loaded_dict is not dict or 'model_state_dict' not in loaded_dict.keys():
            raise UnSupportedFormatError("Format of the loaded model to test is not supported")
        self.model.load_state_dict(loaded_dict['model_state_dict'])
        self.ep_counter = 0

    def test(self, render=False):
        self.ep_counter = 0
        observation = self.env.reset()
        reward_ep = 0

        while self.ep_counter < self.test_ep_count:
            logits, _ = self.model(observation, no_feature_vec=True)

            prob = F.softmax(logits, dim=-1)
            action = prob.multinomial(num_samples=1).detach()

            observation, reward, done, _ = self.env.step(action)
            reward_ep += reward
            if done:
                self.ep_counter += 1
                print("Episode: ", self.ep_counter, " Reward: ", reward_ep)
                reward_ep = 0
                observation = self.env.reset()
            if render:
                self.env.render()
        self.env.close()
