import ray
from agent.worker.rollout_worker import RolloutWorker


@ray.remote
class RayRolloutWorker(RolloutWorker):
    def __init__(self, options_flags, observation_shape, action_count, id, verbose=False):
        super().__init__(options_flags, observation_shape, action_count, id, verbose)
