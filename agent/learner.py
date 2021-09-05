import torch
import torch.nn.functional as F
import threading
import ray
import os
import datetime
import random
import numpy as np
import torch.backends.cudnn
import sys

from rollout_storage.elite_set.buf_population_strategy.brute_force_strategy import BruteForceStrategy
from rollout_storage.elite_set.buf_population_strategy.lim_zero_strategy import LimZeroStrategy
from rollout_storage.elite_set.elite_set_replay import EliteSetReplay
from rollout_storage.experience_replay import ExperienceReplayTorch
from rollout_storage.writer_queue.keep_latest_strategy import KeepLatestStrategy
from stats.data_plotter import create_charts

from stats.nvidia_power_draw import PowerDrawAgent
from utils import compress
from stats.safe_file_writer import SafeFileWriter
from rollout_storage.writer_queue.replay_buffer_writer import ReplayWriterQueue
from torch.optim import Adam
from model.network import ModelNetwork
from queue import Queue
from polynomial_lr_scheduler import PolynomialLRDecay


@ray.remote(num_gpus=torch.cuda.device_count())
class Learner(object):
    def __init__(self, actors, observation_shape, actions_count, options_flags):
        if options_flags.reproducible:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            torch.set_deterministic(True)
            torch.cuda.manual_seed(options_flags.seed)
            torch.cuda.manual_seed_all(options_flags.seed)
            torch.manual_seed(options_flags.seed)
            np.random.seed(options_flags.seed)
            random.seed(options_flags.seed)

        self.options_flags = options_flags
        self.observation_shape = observation_shape
        self.actions_count = actions_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Learner is using CUDA")
        else:
            print("CUDA IS NOT AVAILABLE")
        self.model = ModelNetwork(self.actions_count).to(self.device)

        # self.model.load_state_dict(torch.load('results/BreakoutNoFrameskip-v4_1630692566/regular_model_save_.pt')["model_state_dict"])

        self.optimizer = Adam(self.model.parameters(), lr=self.options_flags.lr)

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=2000, T_mult=1)
        self.lr_scheduler = PolynomialLRDecay(self.optimizer, 6000, 0.000001, 2)


        # size, sample_ratio, options_flags, action_count, env_state_dim, state_compression, feature_vec_dim, insert_strategy : EliteSetInsertStrategy
        self.replay_buffers = [ExperienceReplayTorch(self.options_flags.buffer_size, self.options_flags.replay_data_ratio, self.options_flags, self.actions_count, self.observation_shape),
                               EliteSetReplay(self.options_flags.elite_set_size, self.options_flags.elite_set_data_ratio, self.options_flags, self.actions_count, self.observation_shape,
                                              True, (self.model.get_flatten_layer_output_size(),), BruteForceStrategy(self.options_flags.elite_set_size))]

        self.actors = actors
        self.dict = ray.put({k: v.cpu() for k, v in self.model.state_dict().items()})

    def act(self):
        START_TIME = datetime.datetime.now()
        run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
        os.mkdir("results/" + self.options_flags.env + "_" + str(run_id))
        score_file = SafeFileWriter("results/" + self.options_flags.env + "_" + str(run_id) + "/Scores.txt", "w", 1)
        total_time_file = SafeFileWriter("results/" + self.options_flags.env + "_" + str(run_id) + "/Train_time.txt", "w", 1)
        steps_file = SafeFileWriter("results/" + self.options_flags.env + "_" + str(run_id) + "/Episode_steps.txt", "w", 1)
        loss_file = SafeFileWriter("results/" + self.options_flags.env + "_" + str(run_id) + "/Loss_file.txt", "w", 1)
        lr_file = SafeFileWriter("results/" + self.options_flags.env + "_" + str(run_id) + "/lr_file.txt", "w", 1)
        max_reward_file = SafeFileWriter("results/" + self.options_flags.env + "_" + str(run_id) + "/max_reward_file.txt", "w", 1)
        power_draw_agent = PowerDrawAgent("results/" + self.options_flags.env + "_" + str(run_id) + "/Power_draw.txt", "w", 1)
        score_file.start()
        total_time_file.start()
        steps_file.start()
        loss_file.start()
        lr_file.start()
        max_reward_file.start()
        power_draw_agent.start()

        if not self.options_flags.reproducible:
            rollouts = [actor.performing.remote() for actor in self.actors]
        episode = 0

        learning_lock = threading.Lock()
        batch_lock = threading.Lock()
        actor_lock = threading.Lock()

        loaded_model_warmed_up_event = threading.Event()

        replay_writer = ReplayWriterQueue(self.replay_buffers, queue_size=10, fill_in_strategy=KeepLatestStrategy())
        replay_writer.start()

        stop_event = threading.Event()

        score_queue = Queue(maxsize=100)
        iteration_counter = 0

        max_avg_reward = -sys.maxsize

        training_iteration = 0

        def manage_workers():
            try:
                nonlocal rollouts, episode, iteration_counter, max_avg_reward, score_queue, loss_file, training_iteration, lr_file

                rew_avg = -sys.maxsize
                max_reward = -sys.maxsize
                while episode <= self.options_flags.max_episodes:
                    if stop_event.is_set():
                        break

                    if actor_lock.acquire(False):

                        if self.options_flags.reproducible:
                            done_ref, _ = ray.wait([actor.performing.remote(self.dict, update=True) for actor in self.actors], num_returns=len(self.actors))
                        else:
                            done_ref, rollouts = ray.wait(rollouts, num_returns=1)
                            actor_lock.release()

                        iteration_counter += 1

                        actor_handler = ray.get(done_ref)

                        for j in range(len(done_ref)):
                            worker_buffers, actor_index, rewards, ep_steps = actor_handler[j]

                            episode += len(rewards)

                            score_file.write(rewards)
                            steps_file.write(ep_steps)

                            if len(rewards) > 0:
                                local_max_reward = np.max(rewards)
                                if local_max_reward > max_reward:
                                    max_reward = local_max_reward

                                max_reward_file.write([max_reward for _ in range(len(rewards))])

                            if len(rewards) > 0:
                                total_time_file.write([str(len(rewards)) + ',' + str(datetime.datetime.now() - START_TIME) + ',' + str((datetime.datetime.now() - START_TIME).total_seconds()) + "," + str(training_iteration)])

                            for k in range(len(rewards)):
                                if score_queue.full():
                                    score_queue.get()
                                score_queue.put(rewards[k])

                            if not score_queue.empty():
                                rew_avg = np.average(list(score_queue.queue))
                                if rew_avg > max_avg_reward:
                                    max_avg_reward = rew_avg
                                    print("New MAX average reward per 100/ep: ", max_avg_reward, ' Lr: ', self.lr_scheduler.get_last_lr())
                                    torch.save(
                                        {
                                            "model_state_dict": self.model.state_dict(),
                                            "optimizer_state_dict": self.optimizer.state_dict(),
                                            "scheduler_state_dict": self.lr_scheduler.state_dict(),
                                            "flags": vars(self.options_flags),
                                        },
                                        "results/" + self.options_flags.env + "_" + str(run_id) + '/best_model_save_.pt')
                                    # if rew_avg > 240 and not loaded_model_warmed_up_event.is_set():
                                    #     loaded_model_warmed_up_event.set()

                            if iteration_counter % 50 == 0:
                                print('Episode ', episode, '  Iteration: ', iteration_counter, "  Avg. reward 100/ep: ", rew_avg, " Training iterations: ", training_iteration, ' Lr: ', self.lr_scheduler.get_last_lr())

                            if rew_avg >= 20.2 or training_iteration >= 1000000:
                                torch.save(
                                    {
                                        "model_state_dict": self.model.state_dict(),
                                        "optimizer_state_dict": self.optimizer.state_dict(),
                                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                                        "flags": vars(self.options_flags),
                                    },
                                    "results/" + self.options_flags.env + "_" + str(run_id) + '/best_model_save_.pt')
                                stop_event.set()
                                for p in range(len(self.replay_buffers)):
                                    self.replay_buffers[p].replay_filled_event.set()
                                loaded_model_warmed_up_event.set()
                                break

                            if self.options_flags.reproducible:
                                for i in range(len(worker_buffers)):
                                    for p in range(len(self.replay_buffers)):
                                        self.replay_buffers[p].store_next(state=compress(worker_buffers[i].states),
                                                                          action=worker_buffers[i].actions,
                                                                          reward=worker_buffers[i].rewards,
                                                                          logits=worker_buffers[i].logits,
                                                                          not_done=worker_buffers[i].not_done,
                                                                          feature_vec=worker_buffers[i].feature_vec,
                                                                          random_search=True,
                                                                          add_rew_feature=True,
                                                                          p=2)
                            else:
                                replay_writer.write(worker_buffers)

                            if not self.options_flags.reproducible:
                                with actor_lock:
                                    rollouts.extend([self.actors[actor_index].performing.remote(self.dict, update=True)])

                        if self.options_flags.reproducible:
                            actor_lock.release()

                            # TODO this condition is no complete - need to check the whole array self.replay_buffers
                            if self.replay_buffers[0].filled:

                                self.learning(learning_lock, self.model, self.optimizer, loss_file, self.lr_scheduler, lr_file)

                                self.dict = ray.put({k: v.cpu() for k, v in self.model.state_dict().items()})

                                training_iteration += 1

                        if iteration_counter % 1000 == 0:
                            torch.save(
                                {
                                    "model_state_dict": self.model.state_dict(),
                                    "optimizer_state_dict": self.optimizer.state_dict(),
                                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                                    "flags": vars(self.options_flags),
                                },
                                "results/" + self.options_flags.env + "_" + str(run_id) + '/regular_model_save_.pt')

                    # TODO REVISIT/REFACTOR PERIODICAL ELIT SET FEATURE RECALCULATION
                    # with batch_lock:
                    #     if iteration_counter % 220 == 0 and self.replay_buffer.filled and self.replay_buffer.prior_buf_filled:
                    #         print("recalculating features")
                    #         prior_states = self.replay_buffer.get_prior_buf_states()
                    #
                    #         self.feature_reset_model.load_state_dict({k: v.cpu() for k, v in self.model.state_dict().items()})
                    #
                    #         with torch.no_grad():
                    #             _, _, feature_vecoefs_prior = self.feature_reset_model(prior_states, True)
                    #         self.replay_buffer.set_feature_vecoefs_prior(feature_vecoefs_prior)
                    #
                    #         # feature_vecoefs = None
                    #         # for j in range(6):
                    #         #     with torch.no_grad():
                    #         #         _, _, feature_vecoefs_prior = self.model(prior_states[j*100:(j+1)*100].cuda(), True)
                    #         #     if feature_vecoefs is None:
                    #         #         feature_vecoefs = feature_vecoefs_prior.cpu()
                    #         #     else:
                    #         #         feature_vecoefs = torch.cat((feature_vecoefs, feature_vecoefs_prior.cpu()), 0)
                    #         # self.replay_buffer.set_feature_vecoefs_prior(feature_vecoefs)
                    #
                    #
                    #         del prior_states
                    #         print("recalculating features DONE")

            except Exception as e:
                print("program interrupted learner")
                raise e

            print("ending actor")

        def learning_trd(model, optimizer, lr_scheduler):
            nonlocal loss_file, training_iteration, lr_file
            for p in range(len(self.replay_buffers)):
                self.replay_buffers[p].replay_filled_event.wait()
            # loaded_model_warmed_up_event.wait()

            while training_iteration < self.options_flags.training_max_steps:
                training_iteration += 1
                if stop_event.is_set():
                    break
                self.learning(learning_lock, model, optimizer, loss_file, lr_scheduler, lr_file)
                self.dict = ray.put({k: v.cpu() for k, v in model.state_dict().items()})

            stop_event.set()
            print("Training iterations runs out - end of training")

        threads = []
        thread = threading.Thread(target=manage_workers, name="manage_workers")
        thread.start()
        threads.append(thread)

        if not self.options_flags.reproducible:
            for i in range(3):
                thread = threading.Thread(target=learning_trd, name="learning_trd-%d" % i,
                                          args=(self.model, self.optimizer, self.lr_scheduler))
                thread.start()
                threads.append(thread)

        for thread in threads:
            thread.join()

        score_file.close()
        total_time_file.close()
        steps_file.close()
        loss_file.close()
        lr_file.close()
        max_reward_file.close()
        power_draw_agent.close()
        replay_writer.close()

        create_charts("results/" + self.options_flags.env + "_" + str(run_id))
        return 0

    def learning(self, learning_lock, model_net, optimizer_net, loss_file, lr_scheduler, lr_file):
       
        states, actions, rewards, beh_logits, not_done = self.replay_buffers[0].random_sample(self.options_flags.batch_size)
        for i in range(1, len(self.replay_buffers)):
            states_n, actions_n, rewards_n, beh_logits_n, not_done_n = self.replay_buffers[i].random_sample(
                self.options_flags.batch_size)
            states = torch.cat((states, states_n), 0)
            actions = torch.cat((actions, actions_n), 0)
            rewards = torch.cat((rewards, rewards_n), 0)
            beh_logits = torch.cat((beh_logits, beh_logits_n), 0)
            not_done = torch.cat((not_done, not_done_n), 0)

        states, actions, rewards, beh_logits, not_done = states.to(self.device).transpose(1, 0), actions.to(self.device).transpose(1, 0), rewards.to(self.device).transpose(1, 0), beh_logits.to(self.device).transpose(1, 0), not_done.to(self.device).transpose(1, 0)

        with learning_lock:

            current_logits, current_values = model_net(states.detach(), no_feature_vec=True)
            bootstrap_value = current_values[-1].squeeze(-1)
            current_values = current_values.squeeze(-1)

            target_log_policy = F.log_softmax(current_logits[:-1], dim=-1)
            target_action_log_probs = target_log_policy.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            
            behavior_log_policy = F.log_softmax(beh_logits, dim=-1)
            behavior_action_log_probs = behavior_log_policy.gather(2, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                policy_rate = torch.exp(target_action_log_probs - behavior_action_log_probs)
                rhos = torch.clamp(policy_rate, max=self.options_flags.rho_const)
                coefs = torch.clamp(policy_rate, max=self.options_flags.c_const)

                values_t_plus_1 = current_values[1:]

                discounts = not_done.long() * self.options_flags.gamma

                deltas = rhos * (rewards + discounts * values_t_plus_1 - current_values[:-1])

                vs = torch.zeros((self.options_flags.r_f_steps, self.options_flags.batch_size)).to(self.device)
                
                vs_minus_v_xs = torch.zeros_like(bootstrap_value)
                for t in reversed(range(self.options_flags.r_f_steps)):
                    vs_minus_v_xs = deltas[t] + discounts[t] * coefs[t] * vs_minus_v_xs
                    vs[t] = current_values[t] + vs_minus_v_xs

                vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
                advantages_vt = rhos * (rewards + discounts * vs_t_plus_1 - current_values[:-1])

            policy = F.softmax(current_logits[:-1], dim=-1)
            log_policy = F.log_softmax(current_logits[:-1], dim=-1)
            entropy_loss = self.options_flags.entropy_coef * torch.sum(policy * log_policy)

            baseline_loss = self.options_flags.baseline_loss_coef * torch.sum((vs - current_values[:-1]) ** 2)

            policy_loss = -torch.sum(target_action_log_probs * advantages_vt)

            loss = policy_loss + baseline_loss + entropy_loss

            optimizer_net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_net.parameters(), self.options_flags.max_grad_norm)
            optimizer_net.step()
            lr_scheduler.step()

            loss_file.write([str(policy_loss.item()) + ',' + str(baseline_loss.item()) + ',' + str(entropy_loss.item())])
            lr_file.write([str(lr_scheduler.get_last_lr()[0])])
