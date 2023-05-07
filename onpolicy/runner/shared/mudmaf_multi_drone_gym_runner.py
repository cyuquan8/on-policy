import imageio
import numpy as np
import time
import torch
import wandb

from onpolicy.runner.shared.mudmaf_base_runner import MuDMAFRunner

def _t2n(x):
    return x.detach().cpu().numpy()

class MuDMAFMultiDroneGymRunner(MuDMAFRunner):
    """
    Runner class to perform training, evaluation. and data collection for the multi drone gym.
    """
    def __init__(self, config):
        super(MuDMAFRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # sample actions
                values, actions, action_log_probs, lstm_hidden_states_actor, lstm_cell_states_actor, \
                lstm_hidden_states_critic, lstm_cell_states_critic, actions_env = self.collect(step)
                    
                # observe reward and next obs
                obs, goals, rewards, dones, infos, available_actions = self.envs.step(actions_env)
	
                data = obs, goals, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, lstm_hidden_states_actor, lstm_cell_states_actor, \
                       lstm_hidden_states_critic, lstm_cell_states_critic

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
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, goals, available_actions = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_goals = goals.reshape(self.n_rollout_threads, -1) 
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            share_goals = np.expand_dims(share_goals, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
            share_goals = goals

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_goals[0] = share_goals.copy()
        self.buffer.goals[0] = goals.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        values, actions, action_log_probs, lstm_hidden_states_actor, lstm_cell_states_actor, \
        lstm_hidden_states_critic, lstm_cell_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.share_goals[step]),
                                              np.concatenate(self.buffer.goals[step]),
                                              np.concatenate(self.buffer.lstm_hidden_states_actor[step]),
                                              np.concatenate(self.buffer.lstm_cell_states_actor[step]),
                                              np.concatenate(self.buffer.lstm_hidden_states_critic[step]),
                                              np.concatenate(self.buffer.lstm_cell_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        lstm_hidden_states_actor = np.array(np.split(_t2n(lstm_hidden_states_actor), self.n_rollout_threads))
        lstm_cell_states_actor = np.array(np.split(_t2n(lstm_cell_states_actor), self.n_rollout_threads))
        lstm_hidden_states_critic = np.array(np.split(_t2n(lstm_hidden_states_critic), self.n_rollout_threads))
        lstm_cell_states_critic = np.array(np.split(_t2n(lstm_cell_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, lstm_hidden_states_actor, lstm_cell_states_actor, \
               lstm_hidden_states_critic, lstm_cell_states_critic, actions_env

    def insert(self, data):
        obs, goals, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, lstm_hidden_states_actor, lstm_cell_states_actor, \
        lstm_hidden_states_critic, lstm_cell_states_critic = data

        lstm_hidden_states_actor[dones == True] = \
            np.zeros(((dones == True).sum(), self.lstm_n_layers, self.lstm_hidden_size), dtype=np.float32)
        lstm_cell_states_actor[dones == True] = \
            np.zeros(((dones == True).sum(), self.lstm_n_layers, self.lstm_hidden_size), dtype=np.float32)
        lstm_hidden_states_critic[dones == True] = \
            np.zeros(((dones == True).sum(), self.lstm_n_layers, self.lstm_hidden_size), dtype=np.float32)
        lstm_cell_states_critic[dones == True] = \
            np.zeros(((dones == True).sum(), self.lstm_n_layers, self.lstm_hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_goals = goals.reshape(self.n_rollout_threads, -1) 
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            share_goals = np.expand_dims(share_goals, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
            share_goals = goals

        self.buffer.insert(share_obs, obs, share_goals, goals, lstm_hidden_states_actor, lstm_cell_states_actor, \
                           lstm_hidden_states_critic, lstm_cell_states_critic, actions, action_log_probs, values, \
                           rewards, masks, available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs, eval_goals, eval_available_actions = self.eval_envs.reset()

        eval_lstm_hidden_states_actor = \
            np.zeros((self.n_eval_rollout_threads, self.num_agents, self.lstm_n_layers, self.lstm_hidden_size), 
                     dtype=np.float32)
        eval_lstm_cell_states_actor = \
            np.zeros((self.n_eval_rollout_threads, self.num_agents, self.lstm_n_layers, self.lstm_hidden_size), 
                     dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_lstm_hidden_states_actor, eval_lstm_cell_states_actor = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_goals),
                                        np.concatenate(eval_lstm_hidden_states_actor),
                                        np.concatenate(eval_lstm_cell_states_actor),
                                        np.concatenate(eval_masks),
                                        available_actions=np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_lstm_hidden_states_actor = np.array(np.split(_t2n(eval_lstm_hidden_states_actor), 
                                                              self.n_eval_rollout_threads))
            eval_lstm_cell_states_actor = np.array(np.split(_t2n(eval_lstm_cell_states_actor), 
                                                            self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Observe reward and next obs
            eval_obs, eval_goals, eval_rewards, eval_dones, eval_infos, eval_available_actions = \
                self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_lstm_hidden_states_actor[eval_dones == True] = \
                np.zeros(((eval_dones == True).sum(), self.lstm_n_layers, self.lstm_hidden_size), 
                         dtype=np.float32)
            eval_lstm_cell_states_actor[eval_dones == True] = \
                np.zeros(((eval_dones == True).sum(), self.lstm_n_layers, self.lstm_hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs, goals, available_actions = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            lstm_hidden_states_actor = \
                np.zeros((self.n_rollout_threads, self.num_agents, self.lstm_n_layers, self.lstm_hidden_size), 
                         dtype=np.float32)
            lstm_cell_states_actor = \
                np.zeros((self.n_rollout_threads, self.num_agents, self.lstm_n_layers, self.lstm_hidden_size), 
                         dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, lstm_hidden_states_actor, lstm_cell_states_actor = \
                    self.trainer.policy.act(np.concatenate(obs),
                                            np.concatenate(goals),
                                            np.concatenate(lstm_hidden_states_actor),
                                            np.concatenate(lstm_cell_states_actor),
                                            np.concatenate(masks),
                                            available_actions=np.concatenate(available_actions),
                                            deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                lstm_hidden_states_actor = np.array(np.split(_t2n(lstm_hidden_states_actor), self.n_rollout_threads))
                lstm_cell_states_actor = np.array(np.split(_t2n(lstm_cell_states_actor), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Observe reward and next obs
                obs, goals, rewards, dones, infos, available_actions = envs.step(actions_env)
                episode_rewards.append(rewards)

                lstm_hidden_states_actor[dones == True] = \
                    np.zeros(((dones == True).sum(), self.lstm_n_layers, self.lstm_hidden_size), dtype=np.float32)
                lstm_cell_states_actor[dones == True] = \
                    np.zeros(((dones == True).sum(), self.lstm_n_layers, self.lstm_hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
