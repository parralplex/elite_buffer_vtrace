import os
import ray
import torch

from model.network import ModelNetwork
from wrappers import atari_wrappers

from agent.rollout_worker import RolloutWorker
from agent.learner import Learner
from option_flags import get_flags

os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == '__main__':

    ray.init()

    options_flags = get_flags()

    placeholder_env = atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(options_flags.env, options_flags.seed),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )

    actions_count = placeholder_env.action_space.n
    observation_shape = placeholder_env.observation_space.shape

    # modl = ModelNetwork(actions_count)
    # modl(torch.zeros((1, 4, 84, 84)))

    placeholder_env.close()
    for j in range(1):
        actors = []
        try:
            for i in range(options_flags.actor_count):
                actors.append(RolloutWorker.remote(options_flags, observation_shape, actions_count, i))

            learner = Learner.remote(actors, observation_shape, actions_count, options_flags)
            learner_handle = learner.act.remote()
            ray.wait([learner_handle])

        except Exception as e:
            # TODO log this error into the file
            print("main program crashed, info: " + str(e.args))
