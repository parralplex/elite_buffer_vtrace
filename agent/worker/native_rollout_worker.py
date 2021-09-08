from agent.worker.rollout_worker import RolloutWorker
from queue import Full

# in python multiprocessing processes can be run only on global functions not class methods (class cannot be pickled and sent to process)


def start_worker(id, queue, shared_list, flags, verbose=False):
    rollout_worker = RolloutWorker(id, flags, verbose)

    while shared_list[1]:
        workers_buffers, actor_id, iteration_rewards, iteration_ep_steps = rollout_worker.performing(model_state_dict=shared_list[0], update=True)
        try:
            queue.put([workers_buffers, id, iteration_rewards, iteration_ep_steps], block=False)
        except Full as exp:
            queue.get()
    return


