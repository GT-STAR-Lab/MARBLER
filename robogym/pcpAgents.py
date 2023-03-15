import numpy as np
from enum import Enum
from rps.utilities.graph import *
from .utilities import *

from .PCP_Cont import contPcpEnv
from .PCP_Grid import gridPcpEnv, parser_grid
import yaml
import gym

class Agent:
    def __init__(self, index, sensing_radius, capture_radius):
        self.index = index
        self.sensing_radius = sensing_radius
        self.capture_radius = capture_radius

    def get_observation( self, state_space, agents, continuous_agent = False):
        '''
            each agent's observation-
                poses of all neighbour agents
                checks whether the prey is within sensing radius of the agent
                or whether prey has been found by any of the neighbours
        '''
        #Checks if any prey is in range of the agent and takes the closest if so
        closest_prey = -1
        for p in state_space['prey']:
            in_range, dist = is_close(state_space['poses'], self.index, p, self.sensing_radius)
            if in_range and (dist < closest_prey or closest_prey == -1):
                prey_loc = p.reshape((1,2))[0]
                closest_prey = dist
        if closest_prey == -1:
            prey_loc = [-1,-1]

        if not continuous_agent:
            observation = np.array([*state_space['poses'][:, self.index ][:2], *prey_loc, self.sensing_radius, self.capture_radius])
        else:
            observation = np.array([*state_space['poses'][:, self.index ], *prey_loc, self.sensing_radius, self.capture_radius])
        
        return observation
    
    def __str__(self):
        return f'Capture: {self.prey_caught}\nIndex: {self.index}'

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class PCPAgents:
    def __init__(self, args, policies, type='grid'):
        # Settings

        self.max_episode_steps = args.max_episode_steps
        self.args = args
        self.policies = policies
        self.type=type
        
        self.N_predator = args.predator
        self.N_capture = args.capture
        self.N = self.N_predator + self.N_capture
        self.num_prey = self.args.num_prey

        self._initialize_agents(args)
        if type=='grid':
            self.env = gridPcpEnv.PCPEnv(self, args)
        else:
            self.env = contPcpEnv.PCPEnv(self, args)

    def _initialize_agents(self, args):
        '''
        Initializes all agents and pushes them into a list - self.agents 
        predators first and then capture agents
        '''
        self.agents = []
        if self.type == 'grid':
            radius = (0 if self.args.predator_radius == 0 else (self.args.predator_radius - .5) / self.args.grid_size)
        else:
            radius = args.predator_radius
        for i in range(self.N_predator):
            self.agents.append( Agent(i, radius, 0) )
        
        if self.type == 'grid':
            radius = (0 if self.args.capture_radius == 0 else (self.args.capture_radius - .5) / self.args.grid_size)
        else:
            radius = args.capture_radius
        for i in range(self.N_capture):
            self.agents.append( Agent(i + self.N_predator, 0, radius) )

    def reset(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.episode_steps = 0
        self.prey_locs = []
        self.num_prey = self.args.num_prey
        self.env.reset()
        return [[0]*(6 * (self.args.num_neighbors + 1))] * self.N
        

    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info
        '''
        terminated = False
        self.episode_steps += 1

        updated_state = self.env.step(actions_)
        obs = self.get_observations(updated_state)
        rewards = self.get_rewards(updated_state)
        
        # condition for checking for the whether the episode is terminated
        if self.episode_steps > self.max_episode_steps or updated_state['num_prey'] == 0:
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
        #critic_observations = np.array(self.prey_locs)

        observations = {}
        for agent in self.agents: 
            observations[agent.index] = agent.get_observation(state_space, self.agents, continuous_agent = (self.type != 'grid'))    
        #    critic_observations = np.concatenate((critic_observations, observations[agent.index]))
        
        full_observations = []
        for i, agent in enumerate(self.agents):
            #full_observations[agent.index] = observations[agent.index]
            full_observations.append(observations[agent.index])
            if self.args.delta > 0:
                nbr_indices = delta_disk_neighbors(state_space['poses'],agent.index,self.args.delta)
            else:
                nbr_indices = get_nearest_neighbors(state_space['poses'], agent.index, self.args.num_neighbors)
            for nbr_index in nbr_indices:
                full_observations[i] = np.concatenate( (full_observations[i],observations[nbr_index]) )
        return full_observations#, critic_observations 

    def get_rewards(self, state_space):
        reward = 0 #Fully shared reward, this is a collaborative environment. TODO: is this too sparse?
        if state_space['num_prey'] < self.num_prey:
            reward = self.args.capture_reward * (self.num_prey - state_space['num_prey'])
            self.num_prey = state_space['num_prey']
        reward += state_space['unseen_prey'] * self.args.no_capture_reward
        reward += state_space['num_prey'] * self.args.no_capture_reward

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
        with open('/home/rtorbati/GTClasses/DLM/Project/Heterogeneous-MARL/robogym/PCP_Grid/grid.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        args = objectview(config)
        #self.env = PCPAgents(parser_grid.create_parser().parse_args(), [], type='grid')
        self.env = PCPAgents(args, [], type='grid')
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()
        self.n_agents = self.env.N

    def reset(self):
        # Reset the wrapped environment and modify the initial observation if needed
        # TODO: rn PCPEnv reset is not returning anything, fix this
        observation = self.env.reset()
        return observation

    def step(self, action_n):
        # Execute the given action in the wrapped environment and modify the observation, reward, done and info if needed
        obs_n, reward_n, done_n, info_n = self.env.step(action_n)
        # done signifies termination, info gives additional info for debugging. Can summarize current state
        return tuple(obs_n), reward_n, done_n, info_n
    
    def get_action_space(self):
        return self.env.get_action_space()
    
    def get_observation_space(self):
        return self.env.get_observation_space()