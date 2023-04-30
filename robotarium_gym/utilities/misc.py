import numpy as np
import random
import os
import importlib
import json
import torch
from rps.utilities.misc import *

def is_close( agent_poses, agent_index, prey_loc, sensing_radius):
    agent_pose = agent_poses[:2, agent_index]
    prey_loc = prey_loc.reshape((1,2))[0]
    dist = np.linalg.norm(agent_pose - prey_loc)
    return dist <= sensing_radius, dist

def get_nearest_neighbors(poses, agent, num_neighbors):
    N = poses.shape[1]
    agents = np.arange(N)
    dists = [np.linalg.norm(poses[:2,x]-poses[:2,agent]) for x in agents]
    mins = np.argpartition(dists, num_neighbors+1)
    return np.delete(mins, np.argwhere(mins==agent))[:num_neighbors]


def get_random_vel():
    '''
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second
    '''
    linear_vel = random.uniform(0.05,0.2) 
    #* (1 if random.uniform(-1,1)>0 else -1)
    angular_vel = 0
    # angular_vel = random.uniform(0,10)
    return np.array([linear_vel,angular_vel]).T

def convert_to_robotarium_poses(locations):
    poses = np.array(locations)
    N = poses.shape[0]
    return np.vstack((poses.T, np.zeros(N)))
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def generate_initial_locations(num_locs, width, height, thresh, start_dist=.3, spawn_left = True):
    '''
    generates initial conditions for the robots and prey
    spawns all of them left of thresh if spawn_left is true, spawns them all right if spawn_left is false
    '''
    poses = generate_initial_conditions(num_locs, spacing=start_dist, width=width, height=height)
    if spawn_left:
        for i in range(len(poses[0])):
            poses[0][i] -= (width/2 - thresh)
            poses[2][i] = 0
    else:
        for i in range(len(poses[0])):
            poses[0][i] += (width/2 - thresh)
            poses[2][i] = 0
    return poses

def load_env_and_model(args, module_dir):
    ''' 
    Helper function to load a model from a specified scenario in args
    '''
    if module_dir == "":
        model_config =args.model_config_file
    else:
        model_config = os.path.join(module_dir, "scenarios", args.scenario, "models", args.model_config_file)
    model_config = objectview(json.load(open(model_config)))
    model_config.n_actions = args.n_actions

    if module_dir == "":
        model_weights = torch.load( args.model_file, map_location=torch.device('cpu'))
    else:
        model_weights = torch.load( os.path.join(module_dir,  "scenarios", args.scenario, "models", args.model_file),\
                         map_location=torch.device('cpu'))
    input_dim = model_weights[list(model_weights.keys())[0]].shape[1]

    if module_dir == "":
        actor = importlib.import_module(args.actor_file)
    else:
        actor = importlib.import_module(f'robotarium_gym.utilities.{args.actor_file}')
    actor = getattr(actor, args.actor_class)
    
    model_config.n_agents = args.n_agents
    model = actor(input_dim, model_config)
    model.load_state_dict(model_weights)

    if module_dir == "":
        env_module = importlib.import_module(args.env_file)
    else:
        env_module = importlib.import_module(f'robotarium_gym.scenarios.{args.scenario}.{args.env_file}')
    env_class = getattr(env_module, args.env_class)
    env = env_class(args)

    model_config.shared_reward = args.shared_reward
    return env, model, model_config


def run_env(config, module_dir):
    env, model, model_config = load_env_and_model(config, module_dir)
    obs = np.array(env.reset())
    n_agents = len(obs)

    totalReward = []
    totalSteps = []
    try:
        for i in range(config.episodes):
            episodeReward = 0
            episodeSteps = 0
            hs = np.array([np.zeros((model_config.hidden_dim, )) for i in range(n_agents)])
            for j in range(config.max_episode_steps):      
                if model_config.obs_agent_id: #Appends the agent id if obs_agent_id is true. TODO: support obs_last_action too
                    obs = np.concatenate([obs,np.eye(n_agents)], axis=1)

                #Gets the q values and then the action from the q values
                if 'NS' in config.actor_class:
                    q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs.T))
                else:
                    q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs))
                    
                actions = np.argmax(q_values.detach().numpy(), axis=1)

                obs, reward, done, _ = env.step(actions)
                
                if model_config.shared_reward:
                    episodeReward += reward[0]
                else:
                    episodeReward += sum(reward)
                if done[0]:
                    episodeSteps = j+1
                    break
            if episodeSteps == 0:
                episodeSteps = config.max_episode_steps
            obs = np.array(env.reset())
            print('Episode', i+1)
            print('Episode reward:', episodeReward)
            print('Episode steps:', episodeSteps)
            totalReward.append(episodeReward)
            totalSteps.append(episodeSteps)
    except Exception as error:
        print(error)
    finally:
        print('\nTotal Reward:',totalReward, sum(totalReward))
        print('Total Steps:', totalSteps, sum(totalSteps))