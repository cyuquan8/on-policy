import numpy as np
import time
import torch
import wandb

from functools import reduce
from onpolicy.runner.shared.gcmnet_base_runner import GCMNetRunner

def _t2n(x):
    return x.detach().cpu().numpy()

class GCMNetSMACRunner(GCMNetRunner):
    """
    Runner class to perform training, evaluation. and data collection for SMAC for GCMNet. 
    See parent class for details.
    """
    def __init__(self, config):
        super(GCMNetSMACRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # sample actions
                values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
                scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
                somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic, obs_pred = \
                    self.collect(step)

                # observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
                       scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
                       somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic, obs_pred

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

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x * y, list(
                    self.buffer.active_masks.shape))

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

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        
        value, action, action_log_prob, somu_hidden_states_actor, somu_cell_states_actor, \
        scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
        somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic, obs_pred = \
            self.trainer.policy.get_actions(
                cent_obs=self.buffer.share_obs[step],
                obs=self.buffer.obs[step],
                masks=self.buffer.masks[step],
                available_actions=self.buffer.available_actions[step],
                somu_hidden_states_actor=self.buffer.somu_hidden_states_actor[step] \
                    if self.buffer.somu_hidden_states_actor is not None else None,
                somu_cell_states_actor=self.buffer.somu_cell_states_actor[step] \
                    if self.buffer.somu_cell_states_actor is not None else None,
                scmu_hidden_states_actor=self.buffer.scmu_hidden_states_actor[step] \
                    if self.buffer.scmu_hidden_states_actor is not None else None,
                scmu_cell_states_actor=self.buffer.scmu_cell_states_actor[step] \
                    if self.buffer.scmu_cell_states_actor is not None else None,
                somu_hidden_states_critic=self.buffer.somu_hidden_states_critic[step] \
                    if self.buffer.somu_hidden_states_critic is not None else None,
                somu_cell_states_critic=self.buffer.somu_cell_states_critic[step] \
                    if self.buffer.somu_cell_states_critic is not None else None,
                scmu_hidden_states_critic=self.buffer.scmu_hidden_states_critic[step] \
                    if self.buffer.scmu_hidden_states_critic is not None else None,
                scmu_cell_states_critic=self.buffer.scmu_cell_states_critic[step] \
                    if self.buffer.scmu_cell_states_critic is not None else None
            )

        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        somu_hidden_states_actor = _t2n(somu_hidden_states_actor) if somu_hidden_states_actor is not None else None
        somu_cell_states_actor = _t2n(somu_cell_states_actor) if somu_cell_states_actor is not None else None
        scmu_hidden_states_actor = _t2n(scmu_hidden_states_actor) if scmu_hidden_states_actor is not None else None
        scmu_cell_states_actor = _t2n(scmu_cell_states_actor) if scmu_cell_states_actor is not None else None
        somu_hidden_states_critic = _t2n(somu_hidden_states_critic) if somu_hidden_states_critic is not None else None
        somu_cell_states_critic = _t2n(somu_cell_states_critic) if somu_cell_states_critic is not None else None
        scmu_hidden_states_critic = _t2n(scmu_hidden_states_critic) if scmu_hidden_states_critic is not None else None
        scmu_cell_states_critic = _t2n(scmu_cell_states_critic) if scmu_cell_states_critic is not None else None
        obs_pred = _t2n(obs_pred) if obs_pred is not None else None

        return values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
               scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
               somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic, obs_pred

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
        scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
        somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic, obs_pred = data
        
        dones_env = np.all(dones, axis=1)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if somu_hidden_states_actor is not None:
            somu_hidden_states_actor[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size), 
                         dtype=np.float32)
        if somu_cell_states_actor is not None:
            somu_cell_states_actor[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size), 
                         dtype=np.float32)
        if scmu_hidden_states_actor is not None:
            scmu_hidden_states_actor[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size), 
                         dtype=np.float32)
        if scmu_cell_states_actor is not None:
            scmu_cell_states_actor[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size), 
                         dtype=np.float32)

        if somu_hidden_states_critic is not None:
            somu_hidden_states_critic[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size), 
                         dtype=np.float32)
        if somu_cell_states_critic is not None:
            somu_cell_states_critic[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size), 
                         dtype=np.float32)
        if scmu_hidden_states_critic is not None:
            scmu_hidden_states_critic[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size), 
                         dtype=np.float32)
        if scmu_cell_states_critic is not None:
            scmu_cell_states_critic[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size), 
                         dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs=share_obs, 
            obs=obs, 
            actions=actions, 
            action_log_probs=action_log_probs, 
            value_preds=values, 
            rewards=rewards, 
            masks=masks, 
            bad_masks=bad_masks, 
            active_masks=active_masks, 
            available_actions=available_actions,
            somu_hidden_states_actor=somu_hidden_states_actor, 
            somu_cell_states_actor=somu_cell_states_actor, 
            scmu_hidden_states_actor=scmu_hidden_states_actor,
            scmu_cell_states_actor=scmu_cell_states_actor, 
            somu_hidden_states_critic=somu_hidden_states_critic, 
            somu_cell_states_critic=somu_cell_states_critic, 
            scmu_hidden_states_critic=scmu_hidden_states_critic, 
            scmu_cell_states_critic=scmu_cell_states_critic, 
            obs_pred=obs_pred
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        if self.somu_actor:
            eval_somu_hidden_states_actor = \
                np.zeros((self.n_eval_rollout_threads, self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size), 
                         dtype=np.float32)
            eval_somu_cell_states_actor = \
                np.zeros((self.n_eval_rollout_threads, self.num_agents, self.somu_n_layers, self.somu_lstm_hidden_size), 
                         dtype=np.float32)
        else:
            eval_somu_hidden_states_actor = None
            eval_somu_cell_states_actor = None
        if self.scmu_actor:
            eval_scmu_hidden_states_actor = \
                np.zeros((self.n_eval_rollout_threads, self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size), 
                         dtype=np.float32)

            eval_scmu_cell_states_actor = \
                np.zeros((self.n_eval_rollout_threads, self.num_agents, self.scmu_n_layers, self.scmu_lstm_hidden_size), 
                          dtype=np.float32)
        else:
            eval_scmu_hidden_states_actor = None
            eval_scmu_cell_states_actor = None
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_somu_hidden_states_actor, eval_somu_cell_states_actor, eval_scmu_hidden_states_actor, \
            eval_scmu_cell_states_actor = self.trainer.policy.act(
                obs=eval_obs,
                masks=eval_masks,
                available_actions=eval_available_actions,
                somu_hidden_states_actor=eval_somu_hidden_states_actor,
                somu_cell_states_actor=eval_somu_cell_states_actor,
                scmu_hidden_states_actor=eval_scmu_hidden_states_actor,
                scmu_cell_states_actor=eval_scmu_cell_states_actor,
                deterministic=True
            )
            
            eval_actions = _t2n(eval_actions)
            eval_somu_hidden_states_actor = \
                _t2n(eval_somu_hidden_states_actor) if eval_somu_hidden_states_actor is not None else None
            eval_somu_cell_states_actor = \
                _t2n(eval_somu_cell_states_actor) if eval_somu_cell_states_actor is not None else None
            eval_scmu_hidden_states_actor = \
                _t2n(eval_scmu_hidden_states_actor) if eval_scmu_hidden_states_actor is not None else None
            eval_scmu_cell_states_actor = \
                _t2n(eval_scmu_cell_states_actor) if eval_scmu_cell_states_actor is not None else None

            # Observe reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = \
                self.eval_envs.step(eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            if eval_somu_hidden_states_actor is not None:
                eval_somu_hidden_states_actor[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), 
                               self.num_agents, 
                               self.somu_n_layers, 
                               self.somu_lstm_hidden_size), 
                             dtype=np.float32)
            if eval_somu_cell_states_actor is not None:
                eval_somu_cell_states_actor[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), 
                               self.num_agents, 
                               self.somu_n_layers, 
                               self.somu_lstm_hidden_size), 
                             dtype=np.float32)
            if eval_scmu_hidden_states_actor is not None:
                eval_scmu_hidden_states_actor[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), 
                               self.num_agents, 
                               self.scmu_n_layers, 
                               self.scmu_lstm_hidden_size), 
                             dtype=np.float32)
            if eval_scmu_cell_states_actor is not None:
                eval_scmu_cell_states_actor[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), 
                               self.num_agents, 
                               self.scmu_n_layers, 
                               self.scmu_lstm_hidden_size), 
                             dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = \
                np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
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