import ray
from agent.worker.rollout_worker import RolloutWorker


@ray.remote
class RayRolloutWorker(RolloutWorker):
    def __init__(self, id, flags, verbose=False):
        super().__init__(id, flags, verbose)
