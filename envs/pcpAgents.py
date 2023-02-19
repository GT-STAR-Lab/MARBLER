import argparse
import numpy as np
from enum import Enum
from rps.utilities.graph import *
from utilities import *
from pcpEnv import *

TYPE = Enum('TYPE', ['Predator', 'Capture'])
class Agent:
    def __init__(self, index, agent_type, sensing_radius):
        self.index = index
        self.type = agent_type
        self.sensing_radius = sensing_radius

    def get_observation( self, nbr_indices, state_space):
        '''
            each agent's observation-
                poses of all neighbour agents
                whether the prey is within sensing radius of the agent
        '''
        observation = {}
        # get the poses of all neighbours
        observation['neighbours'] = []
        for nbr_index in nbr_indices:
            observation['neighbours'].append( state_space['poses'][:, nbr_index ] )
        
            # check if prey is within sensing radius of the agent
            observation['prey_found'] = is_close(state_space['poses'][:, self.index ], \
                                state_space['prey'], self.sensing_radius)
        return observation

    # def get_action(self, )

class PCPAgents:
    def __init__(self, args):

        # Settings
        self.max_episode_steps = args.max_episode_steps


        self.N_predator = args.predator
        self.predator_radius = args.predator_radius
        self.N_capture = args.capture
        self.capture_radius = args.capture_radius

        self.N = self.N_predator + self.N_capture

        self._initialize_agents()

        # Laplacian graph considering all agents communicating with each other
        # L = D - A
        # Could change it to a dynamic, sparse graph
        self.Laplacian = completeGL(self.N)

        self.env = PCPEnv(self, args)

    def _initialize_agents(self):
        '''
        Initializes all agents and pushes them into a list - self.agents 
        predators first and then capture agents
        '''
        self.agents = []
        for i in range(self.N_predator):
            self.agents.append( Agent(i, TYPE.Predator, self.predator_radius) )
        for i in range(self.N_capture):
            self.agents.append( Agent(i+self.N_predator, TYPE.Capture, self.capture_radius) )

    def run_episode(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.episode_steps = 0
        self.env.run_episode()

    def get_actions(self, state_space):
        '''
        returns numpy array that is 2XNum_Robots
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second
        Each column represents a different robot
        '''
        if self.episode_steps > self.max_episode_steps:
            return []
        self.episode_steps+=1
        
        observations = self.get_observations(state_space)
        rewards = self.get_rewards(state_space)
        actions = np.zeros((2, self.N))
        actions[0] += .1
        return actions
    
    def get_observations(self, state_space):
        '''
        Input: Takes in the current state space of the environment
        Outputs:
            a dictionary of observations for each agent with agent index as key
        '''
        # get pose and velocity of all neighbours based on laplacian graph
        observations = {}
        for agent in self.agents:
            nbr_indices = topological_neighbors(self.Laplacian, agent.index)
            observations[agent.index] = agent.get_observation(nbr_indices, state_space)
        return observations     

    def get_rewards(self, state_space):
        rewards = np.zeros(self.N)
        #for i in range(len(rewards)):
        #    np.linalg.norm(state_space['prey'], state_space[poses[]])
        return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCPAgents tester')
    parser.add_argument('-predator', type=int, default=2)
    parser.add_argument('-predator_radius', type=float, default = .5)
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = .15)
    parser.add_argument('-show_figure', type=bool, default=True)
    parser.add_argument('-real_time', type=bool, default= False)
    parser.add_argument('-max_episode_steps', type=int, default = 1000)
    args = parser.parse_args()
    
    agents = PCPAgents(args)
    agents.run_episode()
    agents.run_episode()
