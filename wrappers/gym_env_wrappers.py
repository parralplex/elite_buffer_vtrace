import gym
import numpy as np
from pickle import dumps, loads
from collections import namedtuple


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=2):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False

        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (80, 80, 1), dtype=np.uint8)

    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.299 * new_frame[:,:,0] + 0.587*new_frame[:, :, 1] + 0.114 * new_frame[:, :, 2]

        new_frame = new_frame[35:195:2, ::2].reshape(80, 80, 1)
        # new_frame = new_frame[0:211:2, ::2].reshape(105, 80, 1)
        # new_frame = cv2.resize(
        #     new_frame, (80, 80), interpolation=cv2.INTER_AREA
        # ).reshape(80, 80, 1)

        return new_frame.astype(np.uint8)


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0,
                                shape =(self.observation_space.shape[-1],
                                        self.observation_space.shape[0],
                                        self.observation_space.shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(n_steps, axis=0),
                                                env.observation_space.high.repeat(n_steps, axis=0),
                                                dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


ActionResult = namedtuple("action_result", ("snapshot", "observation", "reward", "is_done", "info"))


class WithSnapshots(gym.Wrapper):

    def get_snapshot(self):
        self.env.close()          # terminate popup rendering window
        return dumps(self.env)

    def load_snapshot(self, snapshot):
        assert not hasattr(self, "_monitor") or hasattr(self.env, "_monitor"), "can't backtrack while recording"
        self.env.close()
        self.env = loads(snapshot)

    def get_result(self, snapshot, action):
        self.load_snapshot(snapshot)
        obs, r, done, info = self.step(action)
        new_snapshot = self.get_snapshot()
        return ActionResult(new_snapshot, obs, r, done, info)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


def make_env(env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env = WithSnapshots(env)
    # env = EpisodicLifeEnv(env)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    # env = ScaleFrame(env)
    return env
