import ray
from agent.worker.rollout_worker import RolloutWorker


@ray.remote
class RayRolloutWorker(RolloutWorker):
    def __init__(self, id, verbose=False):
        super().__init__(id, verbose)
