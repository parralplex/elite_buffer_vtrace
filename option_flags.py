import argparse
import os


from wrappers.atari_wrappers import make_stock_atari
from ast import literal_eval

parser = argparse.ArgumentParser(description='Distributed RL platform flags')
# GENERAL
parser.add_argument("--op_mode", default="train", choices=["train", "test", "train_w_load"], help="Operation mode of the application - training, testing, continue training from checkpoint")
parser.add_argument("--load_model_url", default="results/pong_23min/model_save_.pt", help="Url of model_checkpoint that should be loaded.")
parser.add_argument("--load_optimizer_save", default=True, help="Whether optimizer and lr-scheduler state will be loaded from checkpoint save or not")
parser.add_argument("--save_model_period", type=int, default=1000, help="Model is saved every n-th training updates.")
parser.add_argument("--debug", type=bool, default=False, help="Determines whether application should be in debug mode.")

# TRAINING PARAMETER
parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
parser.add_argument('--batch_size', type=int, default=100, help='size of a single batch of data used for training')
parser.add_argument('--r_f_steps', type=int, default=10, help='number of environment steps per 1 unroll / rollout fragment / state sequence')
parser.add_argument('--gradient_clip_by_norm_threshold', type=float, default=40)
parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")

# REPRODUCIBILITY
randomize_seed = int.from_bytes(os.urandom(4), byteorder="little")
parser.add_argument('--seed', type=int, default=randomize_seed)
parser.add_argument('--reproducible', type=bool, default=False, help="Determines whether the experiment should be reproducible")

# ENVIRONMENT
env_name = 'AtlantisNoFrameskip-v4'
parser.add_argument('--env', type=str, default=env_name, help="environment name")

# ALE PREPROCESSING
parser.add_argument('--clip_rewards', type=bool, default=True)
parser.add_argument('--reward_clipping_method',  default="abs_one_sign", choices=["abs_one_sign", "abs_one_clamp", "soft_asymmetric"])
parser.add_argument('--skipped_frames', type=int, default=4)
stacked_frames_default = 4
parser.add_argument('--frames_stacked', type=int, default=stacked_frames_default)
parser.add_argument('--noop_threshold', type=int, default=30)
parser.add_argument('--grayscaling_frames', type=bool, default=True)
parser.add_argument('--episodic_life', type=bool, default=True)
p_width = 84
p_height = 84
default_resolution = str(p_height) + "," + str(p_width)
parser.add_argument('--frame_scale_resolution', default=default_resolution, type=lambda s: [int(item) for item in s.split(',')])

placeholder_env = make_stock_atari(env_name)
actions_count = placeholder_env.action_space.n
placeholder_env.close()

parser.add_argument('--actions_count', type=int, default=actions_count, help='number of available actions in env')
parser.add_argument('--observation_shape', type=int, default=(stacked_frames_default, p_height, p_width), help='dimensions of observations in env')

# PARALLELISM
parser.add_argument('--learner_thread_count', type=int, default=1, help='number of parallel learner threads')
parser.add_argument('--worker_count', type=int, default=7, help='number of workers working in parallel')
parser.add_argument('--envs_per_worker', type=int, default=5, help='number of environments per 1 worker')
parser.add_argument("--multiprocessing_backend", default="ray", choices=["ray", "python_native"], help="Type of mutiprocessing library used in app.")
parser.add_argument("--shared_queue_size", type=int, default=4, help="Size of the queue shared by processes to exchange data - only used with python-native multiprocessing module")

# END OF TRAINING CONDITIONS
parser.add_argument('--environment_max_steps', type=int, default=15000000, help='Length of the experiment in steps.')

# REPLAY BUFFER
parser.add_argument('--replay_writer_queue_size', type=int, default=1, help='how many worker observations(which are to be written to replay buffer) can be stored before some of them have to discarded to save memory')
parser.add_argument("--discarding_strategy", default="keep_latest", choices=["keep_latest", "keep_oldest", "alternating"], help="How replay writer resolves cache queue overflow")
parser.add_argument("--use_replay_compression", type=bool, default=True, help="Determines if samples in replay buffers are compressed or not.")
parser.add_argument("--lz4_compression_level", type=int, default=0)
parser.add_argument("--caching", type=bool, default=True, help="Determines whether per-emptive sample batching should be executed.")
parser.add_argument("--caching_threads", type=int, default=1, help="Number of threads that create sample batches in the background in parallel")
parser.add_argument("--cache_sample_size", type=int, default=1, help="Number batches that should be cached per Manager algorithm interation")
parser.add_argument('--cache_output_buffer_size', type=int, default=3)

# STATISTICS + OUTPUT
parser.add_argument('--avg_buff_size', type=int, default=100, help='number of data used to calculate average score')
parser.add_argument('--verbose_worker_out_int', type=int, default=50, help='interval in which progress data is printed to the console')
parser.add_argument('--verbose_learner_out_int', type=int, default=250, help='interval in which progress data is printed to the console')
parser.add_argument('--basic_verbose', type=bool, default=True, help="Determines whether training progress should be outputed to console ?")
parser.add_argument('--worker_verbose', type=bool, default=False, help="Determines whether worker training progress should be outputed to console ?")
parser.add_argument("--background_save", type=bool, default=False, help="Saves training data in background thread - may affect performance")

# TEST
parser.add_argument("--test_episode_count", type=int, default=1000, help="Number of episodes that should be executed during testing.")
parser.add_argument("--render", type=bool, default=True, help="Should rendering be used during testing ?")

# NETWORK
parser.add_argument("--feature_out_layer_size", type=int, default=512)
parser.add_argument("--use_additional_scaling_FC_layer", type=bool, default=True)

# SCHEDULER + OPTIMIZER
parser.add_argument("--lr_scheduler_steps", type=int, default=10000)
parser.add_argument('--optimizer', default="rmsprop", choices=["rmsprop", "adam"])
parser.add_argument('--rmsprop_eps', type=float, default=0.01)
parser.add_argument('--lr_end_value', type=float, default=0.00001)


def replay_parameter_list(s):
    try:
        data = literal_eval(s)
    except:
        raise argparse.ArgumentTypeError("Unable to parse string into valid python formats")
    return [item for item in data]

# EXPERIENCE REPLAY PARAMETERS
'''
Parameters for each replay used in training are grouped to independent dictionaries. These dictionaries are listed in application parameter 'replay_parameters'.
Currently we know 3 different replay types queue, standard, custom.
Types queue and standard share the same required parameter schema -> can be viewed in /utils/parameter_schema.py.
    These parameters are :   type = <queue, standard>           -> refers to the type of replay buffer used
                             capacity                           -> refers to the size of the replay buffer in unrolls / n-steps state transitions
                             sample_ratio                       -> refers to the sample ratio used in sampling by the replay 
Custom replay type has additional parameters on top of the 3 already mentioned.
    These parameters are :   dist_function = <ln_norm, cos_dist, kl_div>            -> refers to the type of feature vector distance function
                             p                                                      -> refers to the p-norm index of ln_norm vector distance function
                             insert_strategy = <elite_insertion>                    -> refers to the insertion strategy used by replay = THIS PARAMETER IS OPTIONAL
                             sample_strategy = <elite_sampling, attentive_sampling> -> refers to the sampling strategy used by replay = THIS PARAMETER IS OPTIONAL
                             lambda_batch_multiplier                                -> refers to lambda parameter of elite sampling
                             alfa_annealing_factor                                  -> refers to the alfa parameter of elite sampling
                             elite_sampling_strategy = <strategy1, strategy2, strategy3, strategy4>   -> refers to the elite sampling strategy used
                             elite_batch_size                       -> refers to the number of state sequences on which elite insertion is calculated 
'''
parser.add_argument('--replay_parameters',  type=replay_parameter_list, default='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "standard", "capacity": 1000, "sample_ratio": 0.5}]', )
parser.add_argument('--training_fill_in_factor',  type=float, default=0.2, help="Defines percentage of the replay capacity that when reached buffer will signal with EVENT that it is filled and training can start.")


# V-TRACE ALGORITHM PARAMETERS
parser.add_argument('--policy_gradient_loss_weight', type=float, default=1)
parser.add_argument('--value_loss_weight', type=float, default=0.5)
parser.add_argument('--entropy_loss_weight', type=float, default=0.01)
parser.add_argument('--c-const', type=float, default=1.0, help="V-trace c hyperparameter.")
parser.add_argument('--rho-const', type=float, default=1.0, help="V-trace rho hyperparameter.")
    # CLEAR loss functions
parser.add_argument("--use_policy_cloning_loss", type=bool, default=True)
parser.add_argument("--use_value_cloning_loss", type=bool, default=True)
parser.add_argument("--policy_cloning_loss_weight", type=float, default=0.05)
parser.add_argument("--value_cloning_loss_weight", type=float, default=0.005)
    # LASER parameters
parser.add_argument("--use_kl_mask", type=bool, default=False, help="Used for clipping gradients of those state transitions that are far too off-policy, that is"
                                                                    'those which KL divergence between target and behavioral policy logits extends over kl_div_threshold. '
                                                                    'Originates from LASER algorithm - https://arxiv.org/pdf/1909.11583.pdf')
parser.add_argument("--kl_div_threshold", type=float, default=0.3)


flags = parser.parse_args()

# can be used to alter flags during runtime
def change_args(**kwargs):
    global flags
    parser.set_defaults(**kwargs)
    flags = parser.parse_args()
    return flags


def set_defaults(**kwargs):
    parser.set_defaults(**kwargs)





