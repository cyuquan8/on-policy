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
            env = ExploreReward(env)
        if args.include_inspect_reward:
            env = InspectReward(env)
        if args.include_defusal_reward:
            env = DefusalReward(env)
        if args.include_beacon_reward:
            env = BeaconReward(env)
        if args.include_proximity_reward:
            env = ProximityReward(env)
        return env

    if args.region == "all":
        env = DragonEnv(mission_length=args.episode_length, 
                        include_perturbations=args.include_perturbations,
                        obs_wrapper=obs_wrapper)
    elif args.region == 'desert':
        env = DesertEnv(mission_length=args.episode_length, 
                        include_perturbations=args.include_perturbations,
                        obs_wrapper=obs_wrapper)
    elif args.region == 'forest':
        env = ForestEnv(mission_length=args.episode_length, 
                        include_perturbations=args.include_perturbations,
                        obs_wrapper=obs_wrapper)
    elif args.region == 'village':
        env = VillageEnv(mission_length=args.episode_length, 
                         include_perturbations=args.include_perturbations,
                         obs_wrapper=obs_wrapper)
    
    env = reward_shapping_wrapper(env)

    return GymWrapper(env)