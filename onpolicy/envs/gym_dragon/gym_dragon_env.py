from gym_dragon.core.world import Region
from gym_dragon.envs import DesertEnv, DragonEnv, ForestEnv, VillageEnv
from gym_dragon.wrappers import (
    GymWrapper,
    ExploreReward,
    InspectReward,
    DefusalReward,
    BeaconReward,
    ProximityReward,
    Memory,
    EdgeIndex,
    ShowAllAgentLocations,
    ShowAllAgentNodes,
    FullyObservable
)

def GymDragonEnv(args):
    '''
    Creates gym_dragon environment based on configs

    Input:
        episode_length          :   Length of mission. assert same as episode length
        valid_regions           :   Regions to be used (desert, forest, village, all)
        include_perturbations   :   Boolean to include pertubations
        obs_wrapper             :   Observation wrapper function
    '''
    def obs_wrapper(obs):
        if args.include_memory_obs:
            obs = Memory(obs)
        if args.include_edge_index_obs:
            obs = EdgeIndex(obs)
        if args.include_all_agent_locations_obs:
            obs = ShowAllAgentLocations(obs)
        if args.include_all_agent_nodes_obs:
            obs = ShowAllAgentNodes(obs)
        if args.include_full_obs:
            obs = FullyObservable(obs)  
        return obs
    def reward_shapping_wrapper(env):
        if args.include_explore_reward:
            env = ExploreReward(env, weight=args.explore_reward_weight)
        if args.include_inspect_reward:
            env = InspectReward(env, weight=args.inspect_reward_weight)
        if args.include_defusal_reward:
            env = DefusalReward(env, weight=args.defusal_reward_weight)
        if args.include_beacon_reward:
            env = BeaconReward(env, weight=args.beacon_reward_weight)
        if args.include_proximity_reward:
            env = ProximityReward(env, weight=args.proximity_reward_weight)
        return env

    desert_budget_weights = {Region.desert: 
                                {'perturbations': args.budget_weight_desert_perturbations, 
                                 'communications': args.budget_weight_desert_communications, 
                                 'bomb_additonal': args.budget_weight_desert_bomb_additonal
                                }
                            }
    forest_budget_weights = {Region.forest: 
                                {'perturbations': args.budget_weight_forest_perturbations, 
                                 'communications': args.budget_weight_forest_communications, 
                                 'bomb_additonal': args.budget_weight_forest_bomb_additonal
                                }
                            }
    village_budget_weights = {Region.village: 
                                {'perturbations': args.budget_weight_village_perturbations, 
                                 'communications': args.budget_weight_village_communications, 
                                 'bomb_additonal': args.budget_weight_village_bomb_additonal
                                }
                             }  
   
    budget_weights = {**desert_budget_weights,
                      **forest_budget_weights,
                      **village_budget_weights
                     }                       

    if args.region == "all":
        env = DragonEnv(mission_length=args.episode_length,
                        recon_phase_length=args.recon_phase_length,
                        seconds_per_timestep=args.seconds_per_timestep,
                        obs_wrapper=obs_wrapper,
                        budget_weights=budget_weights,
                        color_tools_only=args.color_tools_only,
                        include_fuse_bombs=args.include_fuse_bombs,
                        include_fire_bombs=args.include_fire_bombs,
                        include_chained_bombs=args.include_chained_bombs)
    elif args.region == 'desert':
        env = DesertEnv(mission_length=args.episode_length,
                        recon_phase_length=args.recon_phase_length,
                        seconds_per_timestep=args.seconds_per_timestep,
                        obs_wrapper=obs_wrapper,
                        budget_weights=budget_weights,
                        color_tools_only=args.color_tools_only,
                        include_fuse_bombs=args.include_fuse_bombs,
                        include_fire_bombs=args.include_fire_bombs,
                        include_chained_bombs=args.include_chained_bombs)
    elif args.region == 'forest':
        env = ForestEnv(mission_length=args.episode_length,
                        recon_phase_length=args.recon_phase_length,
                        seconds_per_timestep=args.seconds_per_timestep,
                        obs_wrapper=obs_wrapper,
                        budget_weights=budget_weights,
                        color_tools_only=args.color_tools_only,
                        include_fuse_bombs=args.include_fuse_bombs,
                        include_fire_bombs=args.include_fire_bombs,
                        include_chained_bombs=args.include_chained_bombs)
    elif args.region == 'village':
        env = VillageEnv(mission_length=args.episode_length,
                         recon_phase_length=args.recon_phase_length,
                         seconds_per_timestep=args.seconds_per_timestep,
                         obs_wrapper=obs_wrapper,
                         budget_weights=budget_weights,
                         color_tools_only=args.color_tools_only,
                         include_fuse_bombs=args.include_fuse_bombs,
                         include_fire_bombs=args.include_fire_bombs,
                         include_chained_bombs=args.include_chained_bombs)
    
    env = reward_shapping_wrapper(env)

    return GymWrapper(env)