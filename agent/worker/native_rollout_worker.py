from agent.worker.rollout_worker import RolloutWorker
from queue import Full


# in python multiprocessing processes can be run only on global functions not class methods (class cannot be pickled and sent to process)


def start_worker_sync(id, queue, shared_list, flags, model_loaded_event, sync_barrier, file_save_url, verbose=False):
    rollout_worker = RolloutWorker(id, flags, file_save_url, verbose)

    starting = True
    while shared_list[1]:
        try:
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
            rollout_worker.load_model(shared_list[0])
            workers_buffers, worker_id, iteration_rewards, iteration_ep_steps = rollout_worker.exec_and_eval_rollout()

            queue.put([workers_buffers, id, iteration_rewards, iteration_ep_steps])
        except KeyboardInterrupt:
            print("KeyboardInterrupt exception was raised in worker " + str(id) + " attempting to exit peacefully")
            break

    return


def start_worker_async(id, queue, shared_list, flags, file_save_url, verbose=False):
    rollout_worker = RolloutWorker(id, flags, file_save_url, verbose)

    while shared_list[1]:
        try:
            rollout_worker.load_model(shared_list[0])
            workers_buffers, worker_id, iteration_rewards, iteration_ep_steps = rollout_worker.exec_and_eval_rollout()
            try:
                queue.put([workers_buffers, id, iteration_rewards, iteration_ep_steps], block=False)
            except Full as exp:
                queue.get()
        except KeyboardInterrupt:
            print("KeyboardInterrupt exception was raised in worker " + str(id) + " attempting to exit peacefully")
            break
    return


