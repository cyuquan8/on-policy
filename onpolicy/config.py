import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "ippo", "gcmnet_mappo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
        --wandb_resume_run_id
            [for wandb usage], by default None, to specify run id to resume training.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers (default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    GCMNet parameters:
        --gcmnet_gnn_architecture <str> 
            Architecture for GNN layers in GCMNet, (default: "dna_gatv2")
        --gcmnet_gnn_output_dims
            Hidden Size for GNN layers for actor and critic network, (default: 64)
        --gcmnet_gnn_att_heads <int>
            Number of attention heads for GNN architecture with suitable attention mechanism, (default: 1)
        --gcmnet_gnn_dna_gatv2_multi_att_heads <int>
            Number of attention heads for DNAGATv2 Multi-Head Attention for DNA, (default: 1)
        --gcmnet_gnn_att_concat
            Whether to concatenate or average results from multiple heads for GNN architecture with suitable with attention mechanism, (default: False)
        --gcmnet_cpa_model <str>
            Cardinality Preserved Attention (CPA) model for GNN architecture with suitable attention mechanism, (default: 'f_additive')
        --gcmnet_n_gnn_layers <int>
            Number of GNN layers for GCMNet actor and critic network, (default: 2)
        --gcmnet_n_gnn_fc_layers <int>
            Number of MLP layers for MLP in suitable GNN architecture (GINConv, GAINConv), (default: 2)
        --gcmnet_train_eps
            Whether to train epsilon in suitable GNN architecture (GINConv, GAINConv), (default: False)
        --gcmnet_somu_actor
            Whether to use Self Observation Memory Unit (SOMU) in GCMNet for actor network, (default: False)
        --gcmnet_scmu_actor
            Whether to use Self Communication Memory Unit (SCMU) in GCMNet for actor network, (default: False)
        --gcmnet_somu_critic
            Whether to use Self Observation Memory Unit (SOMU) in GCMNet for critic network, (default: False)
        --gcmnet_scmu_critic
            Whether to use Self Communication Memory Unit (SCMU) in GCMNet for critic network, (default: False)
        --gcmnet_somu_lstm_actor" 
            Whether to use LSTM output of Self Observation Memory Unit (SOMU) in GCMNet for actor network, (default: False)
        --gcmnet_scmu_lstm_actor" 
            Whether to use LSTM output of Self Communication Memory Unit (SCMU) in GCMNet for actor network, (default: False)
        --gcmnet_somu_lstm_critic" 
            Whether to use LSTM output of Self Observation Memory Unit (SOMU) in GCMNet for critic network, (default: False)
        --gcmnet_scmu_lstm_critic" 
            Whether to use LSTM output of Self Communication Memory Unit (SCMU) in GCMNet for critic network, (default: False)
        --gcmnet_somu_att_actor
            Whether to use to use self-attention output from hidden and cell states of Self Observation Memory Unit (SOMU) in GCMNet for actor network, (default: False)
        --gcmnet_scmu_att_actor
            Whether to use to use self-attention output from hidden and cell states of Self Communication Memory Unit (SCMU) in GCMNet for actor network, (default: False)
        --gcmnet_somu_att_critic
            Whether to use to use self-attention output from hidden and cell states of Self Observation Memory Unit (SOMU) in GCMNet for critic network, (default: False)
        --gcmnet_scmu_att_critic
            Whether to use to use self-attention output from hidden and cell states of Self Communication Memory Unit (SCMU) in GCMNet for critic network, (default: False)
        --gcmnet_somu_n_layers <int>
            Number of layers of LSTMs in Self Observation Memory Unit (SOMU) in GCMNet, (default: 2)
        --gcmnet_somu_lstm_hidden_size <int>
            Hidden Size for Self Observation Memory Unit (SOMU) LSTMs in GCMNet, (default: 128)
        --gcmnet_somu_multi_att_n_heads <int>
            Number of Heads for Multi-Attention for SOMU outputs in GCMNet, (default: 2)
        --gcmnet_scmu_n_layers <int>
            Number of layers of LSTMs in Self Communication Memory Unit (SCMU) in GCMNet, (default: 2)
        --gcmnet_scmu_lstm_hidden_size <int>
            Hidden Size for Self Communication Memory Unit (SCMU) LSTMs in GCMNet, (default: 128)
        --gcmnet_scmu_multi_att_n_heads <int>
            Number of Heads for Multi-Attention for SCMU outputs in GCMNet, (default: 2)
        --gcmnet_fc_output_dims <int>
            Hidden Size for MLP layers in GCMNet actor and critic network, (default: 128)
        --gcmnet_n_fc_layers <int>
            Number of MLP layers in GCMNet actor and critic network, (default: 2)
        --gcmnet_knn
            Use K-Nearest Neighbour to generate edge index. If False, use fully connected graph (default: False)
        --gcmnet_k <int>
            Number of Neighbours for K-Nearest Neighbour, (default: 1)
        --gcmnet_rni
            Use Random Node Initialisation (RNI), i.e. append randomly generated vectors to observations in GNN, (default: False)
        --gcmnet_rni_ratio <float>
            Ratio of randomly generated vector in RNI to original observation feature vector, (default: 0.25)
        --gcmnet_dynamics 
            Whether to use dynamics models in GCMNet actor network, (default: False)
        --gcmnet_dynamics_reward 
            Whether to use dynamics models in GCMNet actor network to generate intrinsic exploration reward from disagreement via variance, (default: False)
        --gcmnet_dynamics_fc_output_dims
            Hidden Size for MLP layers in dynamics models in GCMNet actor network, (default: 512)
        --gcmnet_dynamics_n_fc_layers
            Number of MLP layers in dynamics models in GCMNet actor network, (default: 2)
        --gcmnet_dynamics_loss_coef
            Coefficient for dynamics model loss, (default: 0.01)
        --gcmnet_dynamics_reward_coef
            Coefficient for intrinsic exploration reward from disagreement via variance, (default: 1)

    MuDMAF parameters:
        --mudmaf_conv_output_dims <int>
            Output dimensions for convolutions in MuDMAF network (default: 256)
        --mudmaf_n_vgg_conv_layers <int>
            Number of VGG-based convolution layers in MuDMAF network (default: 256)
        --mudmaf_vgg_conv_kernel_size <int>
            Kernel size for convolution in convolution layers in MuDMAF network (default: 3)
        --mudmaf_vgg_maxpool_kernel_size <int>
            Maxpool kernel size for convolution in convolution layers in MuDMAF network (default: 2)
        --mudmaf_n_goal_fc_layers <int>
            Number of MLP layers for goal features (default: 2)
        --mudmaf_n_post_concat_fc_layers <int>
            Number of MLP layers for features post concatenation of observation and goal features (default: 2)
        --mudmaf_lstm_hidden_size <int>
            Hidden size for LSTM in MuDMAF network (default: 512)
        --mudmaf_lstm_n_layers <int>
            Number of layers for LSTM in MuDMAF network (default: 1)
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficient of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mappo', choices=["rmappo", "mappo", "ippo","gcmnet_mappo"])
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")
    parser.add_argument("--wandb_resume_run_id", type=str, default=None, help="[for wandb usage], by default None, to specify run id to resume training")

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=3,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # gcmnet network parameters
    parser.add_argument("--gcmnet_gnn_architecture", type=str, default='dna_gatv2', choices=["dna_gatv2", "gin", "gatv2", "gain"], help="Architecture for GNN layers in GCMNet")
    parser.add_argument("--gcmnet_gnn_output_dims", type=int, default=64, help="Hidden Size for GNN layers for actor and critic network")
    parser.add_argument("--gcmnet_gnn_att_heads", type=int, default=1, help="Number of attention heads for GNN architecture with suitable attention mechanism")
    parser.add_argument("--gcmnet_gnn_dna_gatv2_multi_att_heads", type=int, default=1, help="Number of attention heads for DNAGATv2 Multi-Head Attention for DNA")
    parser.add_argument("--gcmnet_gnn_att_concat", action='store_true', default=False, help="Whether to concatenate or average results from multiple heads for GNN architecture with suitable with attention mechanism")
    parser.add_argument("--gcmnet_gnn_cpa_model", type=str, default='f_additive', choices=["none", "f_additive"], help="Cardinality Preserved Attention (CPA) model for GNN architecture with suitable attention mechanism")
    parser.add_argument("--gcmnet_n_gnn_layers", type=int, default=2, help="Number of GNN layers for GCMNet actor and critic network")
    parser.add_argument("--gcmnet_n_gnn_fc_layers", type=int, default=2, help="Number of MLP layers for MLP in suitable GNN architecture (GINConv, GAINConv)")
    parser.add_argument("--gcmnet_train_eps", action='store_true', default=False, help="Whether to train epsilon in suitable GNN architecture (GINConv, GAINConv)")
    parser.add_argument("--gcmnet_somu_actor", action='store_true', default=False, help="Whether to use Self Observation Memory Unit (SOMU) in GCMNet for actor network")
    parser.add_argument("--gcmnet_scmu_actor", action='store_true', default=False, help="Whether to use Self Communication Memory Unit (SCMU) in GCMNet for actor network")
    parser.add_argument("--gcmnet_somu_critic", action='store_true', default=False, help="Whether to use Self Observation Memory Unit (SOMU) in GCMNet for critic network")
    parser.add_argument("--gcmnet_scmu_critic", action='store_true', default=False, help="Whether to use Self Communication Memory Unit (SCMU) in GCMNet for critic network")
    parser.add_argument("--gcmnet_somu_lstm_actor", action='store_true', default=False, help="Whether to use LSTM output of Self Observation Memory Unit (SOMU) in GCMNet for actor network")
    parser.add_argument("--gcmnet_scmu_lstm_actor", action='store_true', default=False, help="Whether to use LSTM output of Self Communication Memory Unit (SCMU) in GCMNet for actor network")
    parser.add_argument("--gcmnet_somu_lstm_critic", action='store_true', default=False, help="Whether to use LSTM output of Self Observation Memory Unit (SOMU) in GCMNet for critic network")
    parser.add_argument("--gcmnet_scmu_lstm_critic", action='store_true', default=False, help="Whether to use LSTM output of Self Communication Memory Unit (SCMU) in GCMNet for critic network")
    parser.add_argument("--gcmnet_somu_att_actor", action='store_true', default=False, help="Whether to use self-attention output from hidden and cell states of Self Observation Memory Unit (SOMU) in GCMNet for actor network")
    parser.add_argument("--gcmnet_scmu_att_actor", action='store_true', default=False, help="Whether to use self-attention output from hidden and cell states of Self Communication Memory Unit (SCMU) in GCMNet for actor network")
    parser.add_argument("--gcmnet_somu_att_critic", action='store_true', default=False, help="Whether to use self-attention output from hidden and cell states of Self Observation Memory Unit (SOMU) in GCMNet for critic network")
    parser.add_argument("--gcmnet_scmu_att_critic", action='store_true', default=False, help="Whether to use self-attention output from hidden and cell states of Self Communication Memory Unit (SCMU) in GCMNet for critic network")
    parser.add_argument("--gcmnet_somu_n_layers", type=int, default=2, help="Number of layers of LSTMs in Self Observation Memory Unit (SOMU) in GCMNet")
    parser.add_argument("--gcmnet_somu_lstm_hidden_size", type=int, default=128, help="Hidden Size for Self Observation Memory Unit (SOMU) LSTMs in GCMNet")
    parser.add_argument("--gcmnet_somu_multi_att_n_heads", type=int, default=2, help="Number of Heads for Multi-Head Attention for SOMU outputs in GCMNet")
    parser.add_argument("--gcmnet_scmu_n_layers", type=int, default=2, help="Number of layers of LSTMs in Self Communication Memory Unit (SCMU) in GCMNet")
    parser.add_argument("--gcmnet_scmu_lstm_hidden_size", type=int, default=128, help="Hidden Size for Self Communication Memory Unit (SCMU) LSTMs in GCMNet")
    parser.add_argument("--gcmnet_scmu_multi_att_n_heads", type=int, default=2, help="Number of Heads for Multi-Head Attention for SCMU outputs in GCMNet")
    parser.add_argument("--gcmnet_fc_output_dims", type=int, default=128, help="Hidden Size for MLP layers in GCMNet actor and critic network")
    parser.add_argument("--gcmnet_n_fc_layers", type=int, default=2, help="Number of MLP layers in GCMNet actor and critic network")
    parser.add_argument("--gcmnet_knn", action='store_true', default=False, help="Use K-Nearest Neighbour to generate edge index. If False, use fully connected graph")
    parser.add_argument("--gcmnet_k", type=int, default=1, help="Number of Neighbours for K-Nearest Neighbour")
    parser.add_argument("--gcmnet_rni", action='store_true', default=False, help="Use Random Node Initialisation (RNI), i.e. append randomly generated vectors to observations in GNN")
    parser.add_argument("--gcmnet_rni_ratio", type=float, default=0.25, help="Ratio of randomly generated vector in RNI to original observation feature vector")
    parser.add_argument("--gcmnet_dynamics", action='store_true', default=False, help="Whether to use dynamics models in GCMNet actor network")
    parser.add_argument("--gcmnet_dynamics_reward", action='store_true', default=False, help="Whether to use dynamics models in GCMNet actor network to generate intrinsic exploration reward from disagreement via variance")
    parser.add_argument("--gcmnet_dynamics_fc_output_dims", type=int, default=512, help="Hidden Size for MLP layers in dynamics models in GCMNet actor network")
    parser.add_argument("--gcmnet_dynamics_n_fc_layers", type=int, default=2, help="Number of MLP layers in dynamics models in GCMNet actor network")
    parser.add_argument("--gcmnet_dynamics_loss_coef", type=float, default=0.01, help="Coefficient for dynamics model loss")
    parser.add_argument("--gcmnet_dynamics_reward_coef", type=float, default=1, help="Coefficient for intrinsic exploration reward from disagreement via variance")

    # mudmaf network parameters
    parser.add_argument("--mudmaf_conv_output_dims", type=int, default=256, help="Output dimensions for convolutions in MuDMAF network")
    parser.add_argument("--mudmaf_n_vgg_conv_layers", type=int, default=2, help="Number of VGG-based convolution layers in MuDMAF network")
    parser.add_argument("--mudmaf_vgg_conv_kernel_size", type=int, default=3, help="Kernel size for convolution in convolution layers in MuDMAF network")
    parser.add_argument("--mudmaf_vgg_maxpool_kernel_size", type=int, default=2, help="Maxpool kernel size for convolution in convolution layers in MuDMAF network")
    parser.add_argument("--mudmaf_n_goal_fc_layers", type=int, default=2, help="Number of MLP layers for goal features")
    parser.add_argument("--mudmaf_n_post_concat_fc_layers", type=int, default=2, help="Number of MLP layers for features post concatenation of observation and goal features")
    parser.add_argument("--mudmaf_lstm_hidden_size", type=int, default=512, help="Hidden size for LSTM in MuDMAF network")
    parser.add_argument("--mudmaf_lstm_n_layers", type=int, default=1, help="Number of layers for LSTM in MuDMAF network")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help='coefficient of weight decay')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,    
                        default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    return parser
