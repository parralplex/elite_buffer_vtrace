import argparse
import os


from wrappers.atari_wrappers import make_atari

parser = argparse.ArgumentParser(description='Buffered V-trace flags')
# GENERAL
parser.add_argument("--op_mode", default="train", choices=["train", "test", "train_w_load"], help="Operation mode of the application - training, testing, continue training from checkpoint")
parser.add_argument("--load_model_uri", default="results/pong_23min/model_save_.pt", help="Uri of model_checkpoint that may be loaded.")
parser.add_argument("--save_model_period", type=int, default=1000, help="Model is saved every n-th learning iteration.")
parser.add_argument("--debug", type=bool, default=False)

# TRAINING PARAMETER
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=100, help='size of a single batch of data used for training')
parser.add_argument('--r_f_steps', type=int, default=10,
                    help='number of environment steps per 1 rollout fragment of a worker_buf')
parser.add_argument('--baseline_loss_coef', type=float, default=0.5)
parser.add_argument('--entropy_loss_coef', type=float, default=0.01)
parser.add_argument('--max_grad_norm', type=float, default=40)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--c-const', type=float, default=1)
parser.add_argument('--rho-const', type=float, default=1)

# REPRODUCIBILITY
randomize_seed = int.from_bytes(os.urandom(4), byteorder="little")
parser.add_argument('--seed', type=int, default=1) #14372685
parser.add_argument('--reproducible', type=bool, default=False)
parser.add_argument('--replay_out_cache_size', type=int, default=1)
parser.add_argument('--max_cache_pos_pointer', type=int, default=1000000)

# ENVIRONMENT
env_name = 'PongNoFrameskip-v4'
parser.add_argument('--env', type=str, default=env_name)

placeholder_env = make_atari(env_name, randomize_seed, clip_rewards=True, frames_skipped=4)
actions_count = placeholder_env.action_space.n
observation_shape = placeholder_env.observation_space.shape
placeholder_env.close()

parser.add_argument('--actions_count', type=int, default=actions_count, help='number of available actions in env')
parser.add_argument('--observation_shape', type=int, default=observation_shape, help='dimensions of observations in env')
parser.add_argument('--clip_rewards', type=bool, default=True)
parser.add_argument('--skipped_frames', type=int, default=4)


# PARALLELISM
parser.add_argument('--learner_thread_count', type=int, default=1, help='number of parallel learner threads')
parser.add_argument('--worker_count', type=int, default=7, help='number of workers working in parallel')
parser.add_argument('--envs_per_worker', type=int, default=5, help='number of environments per 1 worker')
parser.add_argument("--multiprocessing_backend", default="python_native", choices=["ray", "python_native"], help="Type of mutiprocessing library used in app.")
parser.add_argument("--shared_queue_size", type=int, default=4, help="Size of the queue shared by processes to exchange data - woker_data are sent back to be processed")

# END OF TRAINING CONDITIONS
parser.add_argument('--max_episodes', type=int, default=10000000, help='total number of episodes to be executed')
parser.add_argument('--training_max_steps', type=int, default=10000000, help='total number of training iterations')
parser.add_argument('--max_avg_reward', type=float, default=20.5, help='max avg(100) episode reword at which the training stops.')
parser.add_argument('--max_avg_reward_deviation', type=float, default=0.5)
parser.add_argument('--max_avg_rew_time_accept_deviation', type=int, default=120, help='number of seconds after which app will accept max_avg_reward-max_avg_reward_deviation as a legitimate condition to stop execution')
parser.add_argument('--training_seconds', type=int, default=3600)
# REPLAY BUFFER
parser.add_argument('--use_replay_buffer', type=bool, default=False)
parser.add_argument('--replay_writer_queue_size', type=int, default=1, help='how many worker observations(which are to be written to replay buffer) can be stored before some of them have to discarded to save memory')
parser.add_argument("--discarding_strategy", default="keep_latest", choices=["keep_latest", "keep_oldest", "alternating"], help="How replay writer resolves cache queue overflow")
parser.add_argument("--use_state_compression", type=bool, default=False, help="Determines if the states in replay buffers are compressed or not.")
parser.add_argument("--caching", type=bool, default=False)

parser.add_argument('--replay_buffer_size', type=int, default=1000, help='size of replay buffer')
parser.add_argument('--elite_set_size', type=int, default=0, help='size of elite set of replay buffer')
parser.add_argument('--replay_data_ratio', type=float, default=0.5, help='% number of samples used from normal part of replay buffer when creating batch')
parser.add_argument('--elite_set_data_ratio', type=float, default=0.3, help='% number of samples used from elite set when creating batch')

parser.add_argument('--replay_queue_ratio', type=float, default=0.7)
# parser.add_argument('--replay_queue_size', type=int, default=0)

# CONSOLE OUTPUT
parser.add_argument('--avg_buff_size', type=int, default=100, help='number of data used to calculate average score')
parser.add_argument('--verbose_worker_out_int', type=int, default=50, help='interval in which progress data is printed to the console')
parser.add_argument('--verbose_learner_out_int', type=int, default=250, help='interval in which progress data is printed to the console')
parser.add_argument('--basic_verbose', type=bool, default=True)
parser.add_argument('--worker_verbose', type=bool, default=False)

# ELITE SET
parser.add_argument('--use_elite_set', type=bool, default=True)
parser.add_argument('--p', type=int, default=2, help="p-norm index used while calculation distances between feature vecs in elite set.")

parser.add_argument('--elite_insert_strategy', default="dist_input_filter", choices=["dist_input_filter", "None"])
parser.add_argument("--sample_life", type=int, default=15)
parser.add_argument('--dist_direction', default="lim_zero", choices=["lim_zero", "lim_inf"])
parser.add_argument('--drop_old_samples', type=bool, default=False)

parser.add_argument('--elite_sample_strategy', default="None", choices=["policy_sample", "None"])
parser.add_argument('--sample_dist_direction', default="lim_zero", choices=["lim_zero", "lim_inf"])

# TEST
parser.add_argument("--test_episode_count", type=int, default=1000, help="Number of episodes that should be executed during testing.")
parser.add_argument("--render", type=bool, default=True, help="Should rendering be used during testing ?")

# STATS
parser.add_argument("--background_save", type=bool, default=False, help="Saves training data in background thread - may affect performance")

# NETWORK
parser.add_argument("--feature_out_layer_size", type=int, default=128)

# SCHEDULER
parser.add_argument("--scheduler_steps", type=int, default=15)

# executed after this file is imported - meaning we have access to flags as if it was a singleton class
flags = parser.parse_args()


# can be used to alter flags during runtime
def change_args(**kwargs):
    global flags
    if "rnd_index_chance" in kwargs:
        raise KeyError("rnd_index_chance flag cannot be changed on run_time, because it is part of the decorator which cannot be reinitialized")
    parser.set_defaults(**kwargs)
    flags = parser.parse_args()
    return flags




