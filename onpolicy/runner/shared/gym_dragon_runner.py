import imageio
import numpy as np
import time
import torch
import wandb

from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class GymDragonRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(GymDragonRunner, self).__init__(config)
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
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Observe reward and next obs
                obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
            
                data = obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, rnn_states, rnn_states_critic

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
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # rearrange actions to multiagentdict format for gym_dragon
        actions_env_list = []
        for i in range(self.n_rollout_threads):
            agent_actions_list = {}
            for j in range(self.num_agents):
                agent_actions_list[self.index_to_agent_id[j]] = actions[i, j]
            actions_env_list.append(agent_actions_list)
        actions_env = np.array(actions_env_list)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, \
        rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), 
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), 
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, 
                           masks, available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        episode_length = [0 for _ in range(self.n_eval_rollout_threads)]

        eval_episode_rewards = []
        eval_episode_scores = []
        eval_episode_length = []
        one_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]

        eval_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), 
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # rearrange actions to multiagentdict format for gym_dragon
            eval_actions_env_list = []
            for i in range(self.n_eval_rollout_threads):
                eval_agent_actions_list = {}
                for j in range(self.num_agents):
                    eval_agent_actions_list[self.index_to_agent_id[j]] = eval_actions[i, j]
                eval_actions_env_list.append(eval_agent_actions_list)
            eval_actions_env = np.array(eval_actions_env_list)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = \
                self.eval_envs.step(eval_actions_env)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])
                episode_length[eval_i] += 1

            eval_dones_env = np.all(eval_dones, axis=1)
            
            eval_rnn_states[eval_dones_env == True] = \
                np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), 
                         dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), 
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    eval_episode_scores.append(eval_infos[eval_i][self.index_to_agent_id[0]]['score'])
                    eval_episode_length.append(episode_length[eval_i])
                    episode_length[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_scores = np.array(eval_episode_scores)
                eval_episode_length = np.array(eval_episode_length)
                eval_average_episode_rewards = np.mean(eval_episode_rewards)
                eval_average_episode_scores = np.mean(eval_episode_scores)
                eval_std_episode_scores = np.std(eval_episode_scores)
                eval_average_episode_length = np.mean(eval_episode_length)
                eval_std_episode_length = np.std(eval_episode_length)
                print(f"eval average episode rewards: {eval_average_episode_rewards}")   
                print(f"eval average score: {eval_average_episode_scores}")
                print(f"eval std score: {eval_std_episode_scores}")
                print(f"eval average episode length: {eval_average_episode_length}")
                print(f"eval std episode length: {eval_std_episode_length}")
                if self.use_wandb:
                    wandb.log({"eval_average_episode_rewards": eval_average_episode_rewards}, step=total_num_steps)
                    wandb.log({"eval_average_episode_scores": eval_average_episode_scores}, step=total_num_steps)
                    wandb.log({"eval_std_episode_scores": eval_std_episode_scores}, step=total_num_steps)
                    wandb.log({"eval_average_episode_length": eval_average_episode_length}, step=total_num_steps)
                    wandb.log({"eval_std_episode_length": eval_std_episode_length}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_average_episode_rewards", 
                                             {"eval_average_episode_rewards": eval_average_episode_rewards}, 
                                             total_num_steps)
                    self.writter.add_scalars("eval_average_episode_scores", 
                                             {"eval_average_episode_scores": eval_average_episode_scores}, 
                                             total_num_steps)
                    self.writter.add_scalars("eval_std_episode_scores", 
                                             {"eval_std_episode_scores": eval_std_episode_scores}, 
                                             total_num_steps)
                    self.writter.add_scalars("eval_average_episode_length", 
                                             {"eval_average_episode_length": eval_average_episode_length}, 
                                             total_num_steps)
                    self.writter.add_scalars("eval_std_episode_length", 
                                             {"eval_std_episode_length": eval_std_episode_length}, 
                                             total_num_steps)
                break

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs, available_actions = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), 
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    np.concatenate(available_actions),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                # rearrange actions to multiagentdict format for gym_dragon
                actions_env_list = []
                for i in range(self.n_rollout_threads):
                    agent_actions_list = {}
                    for j in range(self.num_agents):
                        agent_actions_list[self.index_to_agent_id[j]] = actions[i, j]
                    actions_env_list.append(agent_actions_list)
                actions_env = np.array(actions_env_list)

                # Obser reward and next obs
                obs, rewards, dones, infos, available_actions = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), 
                                                     dtype=np.float32)
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
            print(f'episode score is: {infos[0][self.index_to_agent_id[0]]["score"]}')

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
