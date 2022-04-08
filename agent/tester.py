import numpy as np
import torch
from utils.logger import logger
from wrappers import atari_wrappers
from model.network import ModelNetwork
from setuptools_scm import get_version
from option_flags import change_args
from queue import Queue


class Tester(object):
    def __init__(self, test_ep_count, model_save_url, flags):
        self.flags = flags
        self.test_ep_count = test_ep_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Tester is using CUDA")
        else:
            print("CUDA IS NOT AVAILABLE")
        self.env = atari_wrappers.make_test_atari(self.flags.env, self.flags)
        self.model = None
        self.episodic_rewards = Queue()
        self.episodic_steps = Queue()
        try:
            self._load_model_state(model_save_url, self.device)
        except Exception as exp:
            logger.exception("Loaded model is probably not compatible with your network architecture from model/network. " + str(exp))
            raise exp

        self.ep_counter = 0

    def test(self):
        self.ep_counter = 0
        observation = torch.from_numpy(self.env.reset()).float().unsqueeze(0).to(self.device)
        reward_ep = 0
        steps = 0

        while self.ep_counter < self.test_ep_count:
            logits, _, _ = self.model(observation)
            action = torch.argmax(logits, dim=1)

            observation, reward, done, _ = self.env.step(action)
            reward_ep += reward
            steps += 1
            if done:
                self.ep_counter += 1
                print("Episode: ", self.ep_counter, " Reward: ", reward_ep)
                self.episodic_rewards.put(reward_ep)
                self.episodic_steps.put(steps)
                reward_ep = 0
                steps = 0
                observation = self.env.reset()
            observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        self.env.close()
        avg_reward = np.average(self.episodic_rewards.queue)
        avg_steps = np.average(self.episodic_steps.queue)
        return avg_reward, avg_steps

    def _load_model_state(self, url, device):
        try:
            state_dict = torch.load(url + '/checkpoint_data_save.pt', map_location=device)
            self.flags = state_dict["flags"]
            self.model = torch.jit.load(url + '/agent_model_scripted_save.pt').to(device).eval()
        except Exception as exp:
            logger.warning("Error encountered while loading model. ")
            raise exp

        try:
            if get_version() != state_dict["project_version"]:
                logger.warning("Loaded model and project have mismatched versions - project_version:" + get_version() + "  loaded_model_version:" + state_dict["project_version"])
        except Exception as exp:
            logger.warning("We were unable to verify save VERSION - this may cause problems.")