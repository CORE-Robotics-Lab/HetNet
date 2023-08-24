import os
import time
import copy
import pickle
import pathlib

import numpy as np
import torch
import wandb

from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()


class PredatorRunner(Runner):
    """
    Runner class to perform data collection, training, and evaluation and for Predator-Prey, Predator-Capture-Prey,
    and FireCommander domains.

    See parent class for details.
    """

    def __init__(self, config):
        super(PredatorRunner, self).__init__(config)
        print("Initializing runner ...")

    def run(self):
        self.warmup()
        print("Running ... ")

        start = time.time()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        train_episode_length = []
        train_battles_won = 0
        train_battles_game = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_obs, rnn_states_critic = self.collect(step)

                # share obs is the state
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                for i in range(self.n_rollout_threads):
                    if np.all(dones[i]):
                        train_episode_length.append(infos[i][0]['episode_steps'])
                        train_battles_game += 1
                        if infos[i][0]['won']:
                            train_battles_won += 1

                data = obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, \
                    rnn_states, rnn_obs, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # Computer return and update the network
            self.compute()

            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode // 10)

            # save logs
            if episode % self.log_interval == 0 or episode == episodes - 1:
                self.logger.save_stats()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                total_time = (end - start) / 3600  # convert to hours
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                    self.algorithm_name,
                    self.experiment_name,
                    episode,
                    episodes,
                    total_num_steps,
                    self.num_env_steps,
                    int(total_num_steps / (end - start))))

                if self.env_name == "PredatorPrey" or self.env_name == "PredatorCapture" or self.env_name == "FireCommander":

                    train_episode_length_mean = np.mean(np.array(train_episode_length))
                    train_infos['training_ep_length_mean'] = train_episode_length_mean
                    print(f"Training episode length is: {train_episode_length_mean}")

                    train_win_rate = train_battles_won / train_battles_game
                    train_infos['training_win_rate'] = train_win_rate

                    train_battles_game = 0
                    train_battles_won = 0
                    train_episode_length = []
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won'] - last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])

                    last_battles_game = battles_game
                    last_battles_won = battles_won

                self.log_train(train_infos, total_num_steps, total_time)

            # evaluate
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, total_time)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_obs, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_obs[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]))

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_obs = np.array(np.split(_t2n(rnn_obs), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_obs, rnn_states_critic

    def insert(self, data):

        obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, \
            rnn_obs, rnn_states_critic = data
        dones_env = np.all(dones, axis=1)

        if self.all_args.use_LSTM:
            if self.all_args.tensor_obs:
                rnn_states = np.expand_dims(rnn_states, axis=2)
                rnn_obs = np.expand_dims(rnn_obs, axis=2)

                rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents * 2,
                                                          self.recurrent_N, 128), dtype=np.float32)
                rnn_obs[dones_env == True] = np.zeros(((dones_env == True).sum(), self.all_args.num_P * 2,
                                                       self.recurrent_N, 16), dtype=np.float32)
            else:
                rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents * 2,
                                                          self.recurrent_N, self.hidden_size), dtype=np.float32)
                rnn_obs[dones_env == True] = np.zeros(((dones_env == True).sum(), self.all_args.num_P * 2,
                                                       self.recurrent_N, 16), dtype=np.float32)

            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents,
                                                             *self.buffer.rnn_states_critic.shape[3:]),
                                                            dtype=np.float32)

        else:
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
                                                            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_obs, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps, total_time):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
            self.logger.log_stat(k, v, total_num_steps, total_time)

    def log_env(self, env_infos, total_num_steps, total_time):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                self.logger.log_stat(k, np.mean(v), total_num_steps, total_time)

    @torch.no_grad()
    def eval(self, total_num_steps, total_time):

        eval_battles_won = 0
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        eval_ep_lengths = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        if self.all_args.use_LSTM:
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents * 2, 1), dtype=np.float32)
        else:
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(
                    # np.concatenate(eval_share_obs),
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    np.concatenate(eval_available_actions),
                    deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # step through the env
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            if self.all_args.use_LSTM:
                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(),
                                                                    self.num_agents * 2, self.recurrent_N,
                                                                    self.hidden_size),
                                                                   dtype=np.float32)
                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents * 2, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents * 2, 1),
                                                              dtype=np.float32)

            else:
                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents,
                                                                    self.recurrent_N, self.hidden_size),
                                                                   dtype=np.float32)
                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                              dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    eval_ep_lengths.append(eval_infos[eval_i][0]["episode_steps"])
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_lengths = np.array(eval_ep_lengths)
                eval_win_rate = np.array([eval_battles_won / eval_episode])

                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_average_episode_lengths': eval_episode_lengths,
                                  'eval_win_rate': eval_win_rate}

                self.log_env(eval_env_infos, total_num_steps, total_time)

                print(f"Eval win rate is {eval_win_rate}")
                print(f"Eval ep length is {np.mean(eval_episode_lengths)}")

                break

    @torch.no_grad()
    def eval_init_conditions(self,):

        # load initial states for evaluation
        init_states = None
        with open('../../scripts/eval/test_config/{}'.format(self.all_args.eval_config), 'rb') as handle:
            init_states = pickle.load(handle)

        eval_battles_won = 0
        eval_episode = 0
        eval_episode_rewards = []
        eval_ep_lengths = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset(init_states)

        if self.all_args.use_LSTM:
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N,
                                        self.hidden_size),
                                       dtype=np.float32)
            eval_rnn_obs = np.zeros((self.n_eval_rollout_threads, self.all_args.num_P * 2, self.recurrent_N, 16),
                                    dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents * 2, 1), dtype=np.float32)

        else:
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N,
                                        self.hidden_size),
                                       dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)


        while True:
            self.trainer.prep_rollout()

            eval_actions, eval_rnn_states, eval_rnn_obs = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                                  np.concatenate(eval_rnn_states),
                                                                                  np.concatenate(eval_rnn_obs),
                                                                                  np.concatenate(eval_masks),
                                                                                  np.concatenate(eval_available_actions),
                                                                                  deterministic=True)

            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            eval_rnn_obs = np.array(np.split(_t2n(eval_rnn_obs), self.n_rollout_threads))

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions \
                = self.eval_envs.step(eval_actions)

            one_episode_rewards.append(eval_rewards)
            eval_dones_env = np.all(eval_dones, axis=1)

            if self.all_args.use_LSTM:
                eval_rnn_states = np.expand_dims(eval_rnn_states, axis=2)
                eval_rnn_obs = np.expand_dims(eval_rnn_obs, axis=2)
                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents * 2,
                                                                    self.recurrent_N, 128),
                                                                   dtype=np.float32)
                eval_rnn_obs[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.all_args.num_P * 2,
                                                                 self.recurrent_N, 16),
                                                                dtype=np.float32)
                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents * 2, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents * 2, 1),
                                                              dtype=np.float32)
            else:
                raise NotImplementedError

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    eval_ep_lengths.append(eval_infos[eval_i][0]["episode_steps"])
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

                    print('Episode {} completed in: {} steps'.format(len(eval_ep_lengths),
                                                                     eval_infos[eval_i][0]["episode_steps"]) )

            if eval_episode >= len(init_states):
                eval_episode_rewards_mean = np.mean(np.array(eval_episode_rewards))
                eval_episode_lengths_mean = np.mean(np.array(eval_ep_lengths))
                eval_episode_lengths_std_error = np.std(np.array(eval_ep_lengths))/eval_episode

                eval_win_rate = np.array([eval_battles_won / eval_episode])
                eval_env_infos = {'eval__episode_rewards_mean': eval_episode_rewards_mean,
                                  'eval_episode_lengths_mean': eval_episode_lengths_mean,
                                  'eval_episode_lengths_standard_error': eval_episode_lengths_std_error,
                                  'eval_win_rate': eval_win_rate}
                # self.log_env(eval_env_infos)

                print("Eval win rate: {}".format(eval_win_rate))
                print("Eval episode length mean: {}".format(eval_episode_lengths_mean))
                print("Eval episode length standard error: {}".format(eval_episode_lengths_std_error))

                d = dict()
                d['reward'] = eval_episode_rewards
                d['steps_taken'] = eval_ep_lengths

                eval_path = '../../scripts/eval/{}'.format(self.all_args.eval_string)
                pathlib.Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
                print("Saving evaluation results to: {}".format(self.all_args.eval_string))

                torch.save(d, self.all_args.eval_string + '_data.pt')

                exit(0)
                break

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == 'hetgat_mappo':
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states[-1]),
                                                         np.concatenate(self.buffer.rnn_obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.available_actions[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
