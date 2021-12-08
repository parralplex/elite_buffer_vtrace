import ray
from agent.worker.rollout_worker import RolloutWorker


@ray.remote
class RayRolloutWorker(RolloutWorker):
    def __init__(self, worker_id, flags, state_transf_model, file_save_url, verbose=False):
        super().__init__(worker_id, flags, state_transf_model, file_save_url, verbose)
