import numpy as np
from enum import Enum
from rps.utilities.graph import *
from utilities import *

import gridPcpEnv
import yaml
import gym
import os

module_dir = os.path.dirname(__file__)
config_path = os.path.join(module_dir, 'PCP_Grid', 'grid.yaml')

class Agent:
    def __init__(self, index, sensing_radius, capture_radius):
        self.index = index
        self.sensing_radius = sensing_radius
        self.capture_radius = capture_radius

    def get_observation( self, state_space, agents):
        '''
            For each agent's observation-
                Checks for all prey in the range of the current agent
                Returns the closest prey if multiple agents in range
            Returns: [agent_x_pos, agent_y_pos, sensed_prey_x_pose, sensed_prey_y_pose, sensing_radius, capture_radius]
            array of dimension [1, OBS_DIM] 
        '''
        # distance from the closest prey in range
        closest_prey = -1
        # Iterate over all prey
        for p in state_space['prey']:
            # For each prey check if they are in range and get the distance
            in_range, dist = is_close(state_space['poses'], self.index, p, self.sensing_radius)
            # If the prey is in range, check if it is the closest till now
            if in_range and (dist < closest_prey or closest_prey == -1):
                prey_loc = p.reshape((1,2))[0]
                closest_prey = dist
        
        # if no prey found in range
        if closest_prey == -1:
            prey_loc = [-1,-1]
        
        # [agent_x_pos, agent_y_pos, sensed_prey_x_pose, sensed_prey_y_pose, sensing_radius, capture_radius]
        observation = np.array([*state_space['poses'][:, self.index ][:2], *prey_loc, self.sensing_radius, self.capture_radius])
        return observation


class PCPAgents:
    def __init__(self, args):
        # Settings
        self.args = args
        self.N = args.predator + args.capture
        self.num_prey = args.num_prey

        self._initialize_agents(args)
        self.env = gridPcpEnv.PCPEnv(self, args)

    def _initialize_agents(self, args):
        '''
        Initializes all agents and pushes them into a list - self.agents 
        predators first and then capture agents
        '''
        self.agents = []
        # Initialize predator agents
        for i in range(self.args.predator):
            self.agents.append( Agent(i, args.predator_radius, 0) )
        # Initialize capture agents
        for i in range(self.args.capture):
            self.agents.append( Agent(i + self.args.predator, 0, args.capture_radius) )

    def reset(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.episode_steps = 0
        self.prey_locs = []
        self.num_prey = self.args.num_prey
        self.env.reset()
        # TODO: clean the empty observation returning
        return [[0]*(6 * (self.args.num_neighbors + 1))] * self.N
        
    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info (empty dictionary for now)
        '''
        terminated = False
        self.episode_steps += 1

        # call the environment step function and get the updated state
        updated_state = self.env.step(actions_)
        # get the observation and reward from the updated state
        obs           = self.get_observations(updated_state)
        rewards       = self.get_rewards(updated_state)
        
        # condition for checking for the whether the episode is terminated
        if self.episode_steps > self.args.max_episode_steps or \
            updated_state['num_prey'] == 0:
            terminated = True             
        
        return obs, [rewards]*self.N, [terminated]*self.N, [{}]*self.N

    def get_action_space(self):
        return self.env.action_space
    
    def get_observation_space(self):
        return self.env.observation_space

    def get_observations(self, state_space):
        '''
        Input: Takes in the current state space of the environment
        Outputs:
            an array with [agent_x_pos, agent_y_pos, sensed_prey_x_pose, sensed_prey_y_pose, sensing_radius, capture_radius]
            concatenated with the same array for the nearest neighbors based on args.delta or args.num_neighbors

            Also returns a global critic observations which is a list that starts with the true position for every prey agent which is then
            concatenated with the list of observations of each agent
        '''
        if self.prey_locs == []:
            for p in state_space['prey']:
                self.prey_locs = np.concatenate((self.prey_locs, p.reshape((1,2))[0]))
        # iterate over all agents and store the observations for each in a dictionary
        # dictionary uses agent index as key
        observations = {}
        for agent in self.agents: 
            observations[agent.index] = agent.get_observation(state_space, self.agents)    
        
        full_observations = []
        for i, agent in enumerate(self.agents):
            full_observations.append(observations[agent.index])
            
            # For getting neighbors in delta radius. Not being used right now to avoid inconsistent observation dimensions
            # if self.args.delta > 0:
            #     nbr_indices = delta_disk_neighbors(state_space['poses'],agent.index,self.args.delta)

            nbr_indices = get_nearest_neighbors(state_space['poses'], agent.index, self.args.num_neighbors)
            
            # full_observation[i] is of dimension [NUM_NBRS, OBS_DIM]
            for nbr_index in nbr_indices:
                full_observations[i] = np.concatenate( (full_observations[i],observations[nbr_index]) )
        # dimension [NUM_AGENTS, NUM_NBRS, OBS_DIM]
        return full_observations

    def get_rewards(self, state_space):
        # Fully shared reward, this is a collaborative environment.
        reward = 0
        # check if any of the prey were captured by checking the current state_space 
        if state_space['num_prey'] < self.num_prey:
            # reward is proportional to the number of agents captured
            reward = self.args.capture_reward * (self.num_prey - state_space['num_prey'])
            self.num_prey = state_space['num_prey']
        reward += state_space['unseen_prey'] * self.args.no_capture_reward

        return reward
    
    def render(self, mode='human'):
        # Render your environment
        pass


class PCPWrapper(gym.Env):
    def __init__(self):
        """Creates a Gym Wrapper for PCPAgents

        Args:
            env (PCPAgents): A PCPAgents object to wrap in a gym env
        """
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        args = objectview(config)
        self.env = PCPAgents(args)
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()
        self.n_agents = self.env.N

    def reset(self):
        # Reset the wrapped environment and return the initial observation
        observation = self.env.reset()
        return observation

    def step(self, action_n):
        # Execute the given action in the wrapped environment
        obs_n, reward_n, done_n, info_n = self.env.step(action_n)
        return tuple(obs_n), reward_n, done_n, info_n
    
    def get_action_space(self):
        return self.env.get_action_space()
    
    def get_observation_space(self):
        return self.env.get_observation_space()