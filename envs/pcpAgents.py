import numpy as np
from enum import Enum
from rps.utilities.graph import *
from utilities import *

from PCP_Cont import contPcpEnv
from PCP_Grid import gridPcpEnv

class Agent:
    def __init__(self, index, sensing_radius, capture_radius):
        self.index = index
        self.sensing_radius = sensing_radius
        self.capture_radius = capture_radius

    def get_observation( self, state_space, agents, include_velocity = False):
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

        if not include_velocity:
            observation = np.array([*state_space['poses'][:, self.index ][:2], *prey_loc, self.sensing_radius, self.capture_radius])
        return observation
    
    def __str__(self):
        return f'Capture: {self.prey_caught}\nIndex: {self.index}'


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
        self.rewards =[0]
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

    def run_episode(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.episode_steps = 0
        self.prey_locs = []
        self.env.run_episode()
        print("Agent rewards for episode: ", sum(self.rewards))

    def get_actions(self, state_space):
        '''
        returns numpy array that is 2XNum_Robots
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second
        Each column represents a different robot
        ''' 
        if self.episode_steps > self.max_episode_steps:
            return []
        
        #Check if every prey agent has already been captured the prey and ends the episode if they have
        if state_space['num_prey'] == 0:
            return [], []
        
        self.episode_steps+=1
        self.observations, critic_observations = self.get_observations(state_space)
        actions = []
        for i in range(self.N):
            actions.append(self.policies[i].getAction(self.observations[i], critic_observations, self.rewards[-1]))
        self.rewards.append(self.get_rewards(state_space, actions))
        return actions, self.agents
    
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
        critic_observations = np.array(self.prey_locs)

        observations = {}
        for agent in self.agents:          
            observations[agent.index] = agent.get_observation(state_space, self.agents)
            critic_observations = np.concatenate((critic_observations, observations[agent.index]))
        
        full_observations = {}
        for agent in self.agents:
            full_observations[agent.index] = observations[agent.index]
            if self.args.delta > 0:
                nbr_indices = delta_disk_neighbors(state_space['poses'],agent.index,self.args.delta)
            else:
                nbr_indices = get_nearest_neighbors(state_space['poses'], agent.index, self.args.num_neighbors)
            for nbr_index in nbr_indices:
                full_observations[agent.index] = np.concatenate( (full_observations[agent.index],observations[nbr_index]) )
        print(critic_observations)
        return full_observations, critic_observations 

    def get_rewards(self, state_space, actions):
        reward = 0 #Fully shared reward, this is a collaborative environment
        if state_space['num_prey'] < self.num_prey:
            reward = self.args.capture_reward * (self.num_prey - state_space['num_prey'])
            self.num_prey = state_space['num_prey']
        else:
            reward = self.args.no_capture_reward

        return reward



