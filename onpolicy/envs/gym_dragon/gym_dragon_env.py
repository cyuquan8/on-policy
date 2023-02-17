from gym_dragon.envs import DesertEnv, DragonEnv, ForestEnv, VillageEnv
from gym_dragon.wrappers import FullyObservable

def obs_wrapper(obs):
    obs = FullyObservable(obs)
    return obs

def GymDragonEnv(args):
    '''
    Creates gym_dragon environment based on configs

    Input:
        episode_length          :   Length of mission. assert same as episode length
        valid_regions           :   Regions to be used (desert, forest, village, all)
        include_perturbations   :   Boolean to include pertubations
        obs_wrapper             :   Observation wrapper function
    '''
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

    return env