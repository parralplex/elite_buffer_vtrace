import argparse
import os

from wrappers.atari_wrappers import make_atari, wrap_pytorch, wrap_deepmind

parser = argparse.ArgumentParser(description='Buffered V-trace flags')
# GENERAL
parser.add_argument("--op_mode", default="train", choices=["train", "test", "train_w_load"], help="Operation mode of the application - training, testing, continue training from checkpoint")
parser.add_argument("--load_model_uri", default="", help="Uri of model_checkpoint that may be loaded.")
parser.add_argument("--save_model_period", type=int, default=1000, help="Model is saved every n-th learning iteration.")

# TRAINING PARAMETER
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
parser.add_argument('--batch_size', type=int, default=40, help='size of a single batch of data used for training')
parser.add_argument('--r_f_steps', type=int, default=50,
                    help='number of environment steps per 1 rollout fragment of a worker_buf')
parser.add_argument('--baseline_loss_coef', type=float, default=0.5)
parser.add_argument('--entropy_loss_coef', type=float, default=0.0005)
parser.add_argument('--max_grad_norm', type=float, default=40)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--c-const', type=float, default=1)
parser.add_argument('--rho-const', type=float, default=1)

# REPRODUCIBILITY
randomize_seed = int.from_bytes(os.urandom(4), byteorder="little")
parser.add_argument('--seed', type=int, default=randomize_seed)
parser.add_argument('--reproducible', type=bool, default=False)

# ENVIRONMENT
env_name = 'PongNoFrameskip-v4'
parser.add_argument('--env', type=str, default=env_name)

placeholder_env = wrap_pytorch(wrap_deepmind(make_atari(env_name, randomize_seed), episode_life=True, clip_rewards=False, frame_stack=True, scale=False))
actions_count = placeholder_env.action_space.n
observation_shape = placeholder_env.observation_space.shape
placeholder_env.close()

parser.add_argument('--actions_count', type=int, default=actions_count, help='number of available actions in env')
parser.add_argument('--observation_shape', type=int, default=observation_shape, help='dimensions of observations in env')

# PARALLELISM
parser.add_argument('--learner_thread_count', type=int, default=3, help='number of parallel learner threads')
parser.add_argument('--worker_count', type=int, default=8, help='number of workers working in parallel')
parser.add_argument('--envs_per_worker', type=int, default=20, help='number of environments per 1 worker')
parser.add_argument("--multiprocessing_backend", default="ray", choices=["ray", "python_native"], help="Type of mutiprocessing library used in app.")
parser.add_argument("--shared_queue_size", type=int, default=10, help="Size of the queue shared by processes to exchange data - woker_data are sent back to be processed")

# END OF TRAINING CONDITIONS
parser.add_argument('--max_episodes', type=int, default=10000000, help='total number of episodes to be executed')
parser.add_argument('--training_max_steps', type=int, default=10000000, help='total number of training iterations')
parser.add_argument('--max_avg_reward', type=float, default=20.5, help='max avg(100) episode reword at which the training stops.')

# REPLAY BUFFER
parser.add_argument('--replay_writer_cache_size', type=int, default=10, help='how many worker observations(which are to be written to replay buffer) can be stored before some of them have to discarded to save memory')
parser.add_argument("--discarding_strategy", default="keep_latest", choices=["keep_latest", "keep_oldest", "alternating"], help="How replay writer resolves cache queue overflow")
parser.add_argument("--use_state_compression", type=bool, default=True, help="Determines if the states in replay buffers are compressed or not.")

parser.add_argument('--replay_buffer_size', type=int, default=900, help='size of replay buffer')
parser.add_argument('--elite_set_size', type=int, default=100, help='size of elite set of replay buffer')
parser.add_argument('--replay_data_ratio', type=float, default=0.9, help='% number of samples used from normal part of replay buffer when creating batch')
parser.add_argument('--elite_set_data_ratio', type=float, default=0.1, help='% number of samples used from elite set when creating batch')

# CONSOLE OUTPUT
parser.add_argument('--avg_buff_size', type=int, default=100, help='number of data used to calculate average score')
parser.add_argument('--verbose_output_interval', type=int, default=50, help='interval in which progress data is printed to the console')

# ELITE SET
parser.add_argument('--use_elite_set', type=bool, default=False)
parser.add_argument("--elite_reset_period", type=int, default=200, help="Elite set features recalculation period.")
parser.add_argument('--random_search', type=bool, default=True, help="Elite set processing(on insert) starts at random index.")
parser.add_argument('--add_rew_feature', type=bool, default=True, help="feature vecs valus are extended by the sum of rewards")
parser.add_argument('--p', type=int, default=2, help="p-norm index used while calculation distances between feature vecs in elite set.")
parser.add_argument('--elite_pop_strategy', default="lim_zero", choices=["lim_zero", "lim_inf", "brute_force"], help="Strategy used to calculate best possible place(index) in elite set for new observation")
parser.add_argument('--rnd_index_chance', type=float, default=0, help="Chance that observation will be inserted into elite set on random index regardless of the population strategy selected")

# TEST
parser.add_argument("--test_episode_count", type=int, default=1000, help="Number of episodes that should be executed during testing.")
parser.add_argument("--render", type=bool, default=True, help="Should rendering be used during testing ?")


# executed after this file is imported - meaning we have access to flags as if it was a singleton class
flags = parser.parse_args()


# can be used to alter flags during runtime
def change_args(**kwargs):
    global flags
    parser.set_defaults(**kwargs)
    flags = parser.parse_args()


def set_flags(new_flags):
    global flags
    flags = new_flags



