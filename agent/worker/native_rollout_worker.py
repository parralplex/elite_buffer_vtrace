from agent.worker.rollout_worker import RolloutWorker
from queue import Full


# in python multiprocessing processes can be run only on global functions not class methods (class cannot be pickled and sent to process)


def start_worker(id, queue, shared_list, flags, model_loaded_event, sync_barrier, verbose=False):
    rollout_worker = RolloutWorker(id, flags, verbose)

    starting = True
    while shared_list[1]:
        if flags.reproducible:
            if not starting:
                if flags.worker_count == 1 or sync_barrier.n_waiting == (flags.worker_count - 1):
                    model_loaded_event.clear()
            else:
                starting = False
            i = sync_barrier.wait()
            if i == 1:
                sync_barrier.reset()
            model_loaded_event.wait()
            if not shared_list[1]:
                break
        workers_buffers, worker_id, iteration_rewards, iteration_ep_steps = rollout_worker.performing(model_state_dict=shared_list[0], update=True)

        if flags.reproducible:
            queue.put([workers_buffers, id, iteration_rewards, iteration_ep_steps])
        else:
            try:
                queue.put([workers_buffers, id, iteration_rewards, iteration_ep_steps], block=False)
            except Full as exp:
                queue.get()
    return


