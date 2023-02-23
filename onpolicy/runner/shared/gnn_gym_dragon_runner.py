import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.gnn_base_runner import GNNRunner

def _t2n(x):
    return x.detach().cpu().numpy()


class GNNGymDragonRunner(GNNRunner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(GNNGymDragonRunner, self).__init__(config)
        self.index_to_agent_id = {0: 'alpha', 1: 'bravo', 2: 'charlie'}

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
                scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
                somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
                       scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
                       somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic

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
                print("\n Region {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.region,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "gym_dragon":
                    # env_infos = {}
                    sum_score = 0
                    for i in range(self.n_rollout_threads):
                        sum_score += infos[i][self.index_to_agent_id[0]]['score']
                    avg_score = round(sum_score / self.n_rollout_threads)
                    print("average score is {}.".format(avg_score))
                    if self.use_wandb:
                        wandb.log({"average score": avg_score}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("average score", {"average score": avg_score}, total_num_steps)

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, available_actions= self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        
        value, action, action_log_prob, somu_hidden_states_actor, somu_cell_states_actor, \
        scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
        somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic \
        = self.trainer.policy.get_actions(self.buffer.share_obs[step],
                                          self.buffer.obs[step],
                                          self.buffer.somu_hidden_states_actor[step],
                                          self.buffer.somu_cell_states_actor[step],
                                          self.buffer.scmu_hidden_states_actor[step],
                                          self.buffer.scmu_cell_states_actor[step],
                                          self.buffer.somu_hidden_states_critic[step],
                                          self.buffer.somu_cell_states_critic[step],
                                          self.buffer.scmu_hidden_states_critic[step],
                                          self.buffer.scmu_cell_states_critic[step],
                                          self.buffer.masks[step],
                                          self.buffer.available_actions[step]
                                          )

        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        somu_hidden_states_actor = _t2n(somu_hidden_states_actor)
        somu_cell_states_actor = _t2n(somu_cell_states_actor)
        scmu_hidden_states_actor = _t2n(scmu_hidden_states_actor)
        scmu_cell_states_actor = _t2n(scmu_cell_states_actor)
        somu_hidden_states_critic = _t2n(somu_hidden_states_critic)
        somu_cell_states_critic = _t2n(somu_cell_states_critic)
        scmu_hidden_states_critic = _t2n(scmu_hidden_states_critic)
        scmu_cell_states_critic = _t2n(scmu_cell_states_critic)

        # rearrange actions to multiagentdict format for gym_dragon
        actions_env_list = []
        for i in range(self.n_rollout_threads):
            agent_actions_list = {}
            for j in range(self.num_agents):
                agent_actions_list[self.index_to_agent_id[j]] = actions[i, j]
            actions_env_list.append(agent_actions_list)
        actions_env = np.array(actions_env_list)

        return values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
               scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
               somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, somu_hidden_states_actor, somu_cell_states_actor, \
        scmu_hidden_states_actor, scmu_cell_states_actor, somu_hidden_states_critic, \
        somu_cell_states_critic, scmu_hidden_states_critic, scmu_cell_states_critic = data
        
        dones_env = np.all(dones, axis=1)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        somu_hidden_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.somu_hidden_states_actor.shape[3:]), dtype=np.float32)
        somu_cell_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.somu_cell_states_actor.shape[3:]), dtype=np.float32)
        scmu_hidden_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.scmu_hidden_states_actor.shape[3:]), dtype=np.float32)
        scmu_cell_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.scmu_cell_states_actor.shape[3:]), dtype=np.float32)

        somu_hidden_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.somu_hidden_states_critic.shape[3:]), dtype=np.float32)
        somu_cell_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.somu_cell_states_critic.shape[3:]), dtype=np.float32)
        scmu_hidden_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.scmu_hidden_states_critic.shape[3:]), dtype=np.float32)
        scmu_cell_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.scmu_cell_states_critic.shape[3:]), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, \
                           scmu_cell_states_actor, somu_hidden_states_critic, somu_cell_states_critic, scmu_hidden_states_critic, \
                           scmu_cell_states_critic, actions, action_log_probs, values, rewards, masks, \
                           available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_somu_hidden_states_actor = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.somu_num_layers, self.somu_lstm_hidden_size), dtype=np.float32)
        eval_somu_cell_states_actor = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.somu_num_layers, self.somu_lstm_hidden_size), dtype=np.float32)
        eval_scmu_hidden_states_actor = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.scmu_num_layers, self.scmu_lstm_hidden_size), dtype=np.float32)
        eval_scmu_cell_states_actor = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.scmu_num_layers, self.scmu_lstm_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_actions, eval_somu_hidden_states_actor, eval_somu_cell_states_actor, eval_scmu_hidden_states_actor, eval_scmu_cell_states_actor = \
                self.trainer.policy.act(eval_obs,
                                        eval_somu_hidden_states_actor,
                                        eval_somu_cell_states_actor,
                                        eval_scmu_hidden_states_actor,
                                        eval_scmu_cell_states_actor,
                                        eval_masks,
                                        eval_available_actions,
                                        deterministic=True
                                        )
            
            eval_actions = _t2n(eval_actions)
            eval_somu_hidden_states_actor = _t2n(eval_somu_hidden_states_actor)
            eval_somu_cell_states_actor = _t2n(eval_somu_cell_states_actor)
            eval_scmu_hidden_states_actor = _t2n(eval_scmu_hidden_states_actor)
            eval_scmu_cell_states_actor = _t2n(eval_scmu_cell_states_actor)

            # rearrange actions to multiagentdict format for gym_dragon
            eval_actions_env_list = []
            for i in range(self.n_eval_rollout_threads):
                eval_agent_actions_list = {}
                for j in range(self.num_agents):
                    eval_agent_actions_list[self.index_to_agent_id[j]] = eval_actions[i, j]
                eval_actions_env_list.append(eval_agent_actions_list)
            eval_actions_env = np.array(eval_actions_env_list)

            # Observe reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_somu_hidden_states_actor[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.somu_num_layers, self.somu_lstm_hidden_size), dtype=np.float32)
            eval_somu_cell_states_actor[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.somu_num_layers, self.somu_lstm_hidden_size), dtype=np.float32)
            eval_scmu_hidden_states_actor[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.scmu_num_layers, self.scmu_lstm_hidden_size), dtype=np.float32)
            eval_scmu_cell_states_actor[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.scmu_num_layers, self.scmu_lstm_hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        eval_sum_score = 0
        for i in range(self.n_eval_rollout_threads):
            eval_sum_score += eval_infos[i][self.index_to_agent_id[0]]['score']
        eval_avg_score = round(eval_sum_score / self.n_eval_rollout_threads)
        print("eval average score is {}.".format(eval_avg_score))
        if self.use_wandb:
            wandb.log({"eval average score": eval_avg_score}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval average score", {"eval average score": avg_score}, total_num_steps)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)