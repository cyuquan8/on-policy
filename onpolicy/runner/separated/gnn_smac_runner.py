import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.separated.base_runner import Runner
from onpolicy.utils.separated_gnn_buffer import SeparatedGnnReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class GNNSMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(GNNSMACRunner, self).__init__(config)

        # overwrite buffer
        self.buffer = []

        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedGnnReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
        from onpolicy.algorithms.dgcn_mappo.dgcn_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.dgcn_mappo.algorithm.dgcnMAPPOPolicy import R_MAPPOPolicy as Policy

        # overwrite policy
        self.policy = []
        self.trainer = []
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)

            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)
            self.trainer.append(tr)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                if self.use_linear_lr_decay:
                    for agent_id in range(self.num_agents):
                        self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.map_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
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
                            incre_battles_game.append(info[0]['battles_game'] - last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(
                        incre_battles_game) > 0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                    last_battles_game = battles_game
                    last_battles_won = battles_won
                for agent_id in range(self.num_agents):
                    train_infos[agent_id]['dead_ratio'] = 1 - self.buffer[agent_id].active_masks.sum() / reduce(lambda x, y: x * y, list(
                    self.buffer[agent_id].active_masks.shape))

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(len(self.num_agents)):
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = obs.copy()
            self.buffer[agent_id].available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        action_log_probs = []
        for agent_id in range(len(self.num_agents)):
            self.trainer[agent_id].prep_rollout()

            value, action, action_log_prob, \
                = self.trainer[agent_id].policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                                  np.concatenate(self.buffer.obs[step]),
                                                  np.concatenate(self.buffer.masks[step]),
                                                  np.concatenate(self.buffer.available_actions[step]))
            # [self.envs, agents, dim]
            values.append(np.array(np.split(_t2n(value), self.n_rollout_threads)))
            actions.append(np.array(np.split(_t2n(action), self.n_rollout_threads)))
            action_log_probs.append(np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads)))


        return values, actions, action_log_probs

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs = data

        dones_env = np.all(dones, axis=1)


        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(np.array(list(share_obs[:, agent_id])),list(obs[:, agent_id]),
            actions[:, agent_id],
            action_log_probs[:, agent_id],
            values[:, agent_id],
            rewards[:, agent_id],
            masks[:, agent_id],
            bad_masks[:, agent_id], active_masks[:, agent_id],
            available_actions[:, agent_id]
            )
            # self.buffer.insert(share_obs, obs,
            #                actions, action_log_probs, values, rewards, masks, bad_masks, active_masks,
            #                available_actions)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()


        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer.prep_rollout()
                eval_actions = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                    eval_actions)
                one_episode_rewards.append(eval_rewards)

                eval_dones_env = np.all(eval_dones, axis=1)


                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                              dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break