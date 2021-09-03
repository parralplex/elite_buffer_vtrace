import argparse
import os

parser = argparse.ArgumentParser(description='Buffered V-trace flags')

parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')

parser.add_argument('--batch-size', type=int, default=40, help='size of a single batch of data used for training')


parser.add_argument('--max-grad-norm', type=float, default=40)
parser.add_argument('--baseline-loss-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.0005)


parser.add_argument('--r_f_steps', type=int, default=50, help='number of environment steps per 1 rollout fragment of a worker')


parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--c-const', type=float, default=1)
parser.add_argument('--rho-const', type=float, default=1)

parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')

randomize_seed = int.from_bytes(os.urandom(4), byteorder="little")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--reproducible', type=bool, default=False)

parser.add_argument('--actor_count', type=int, default=8, help='number of actors working in parallel')
parser.add_argument('--envs-per-actor', type=int, default=20, help='number of environments per 1 actor')

parser.add_argument('--max-episodes', type=int, default=10000000, help='total number of episodes to be executed')
parser.add_argument('--training-max-steps', type=int, default=1000000, help='total number of training episodes')

# ray
parser.add_argument('--wait-task-count', type=int, default=5, help='number of tasks/iterations that are being waited for before processing data ')
parser.add_argument('--actor-update-freq', type=int, default=1, help='number of learning iterations between updating actor networks with new weights')
parser.add_argument('--learner-thread-count', type=int, default=1, help='number of parallel learner threads')

# replay buffer
parser.add_argument('--buffer-size', type=int, default=1000, help='size of replay buffer')
parser.add_argument('--elite-set-size', type=int, default=0, help='size of elite set of replay buffer')
parser.add_argument('--replay-data-ratio', type=float, default=1, help='% number of samples used from normal part of replay buffer when creating batch')
parser.add_argument('--elite-set-data-ratio', type=float, default=0, help='% number of samples used from elite set when creating batch')

# console ouput
parser.add_argument('--avg-buff-size', type=int, default=100, help='number of data used to calculate average score')


def get_flags():
    return parser.parse_args()

