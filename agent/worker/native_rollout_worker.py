from agent.worker.rollout_worker import RolloutWorker

# in python multiprocessing processes can be run only on global functions not class methods (class cannot be pickled and sent to process)


def start_worker(id, queue, shared_list, stop_event, verbose=False):
    rollout_worker = RolloutWorker(id, verbose)

    while not stop_event.is_set():
        workers_buffers, actor_id, iteration_rewards, iteration_ep_steps = rollout_worker.performing(model_state_dict=shared_list[0], update=True)
            # if queue.s >= 10:
            #     queue.get()   TODO implement queue related overflow protection
        queue.put([workers_buffers, id, iteration_rewards, iteration_ep_steps])

