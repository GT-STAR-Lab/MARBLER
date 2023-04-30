import numpy as np
from gym import spaces
import copy
import yaml
import os

#This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from robotarium_gym.utilities.roboEnv import roboEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.Simple.visualize import *
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_gym.scenarios.Simple.agent import Agent

# An extremely simple environment for debugging the policy. 
# It consists of multiple agent who already knows the prey's location 
# and will get dense rewards.

class simple(BaseEnv):
    def __init__(self, args):
        # Settings
        self.args = args

        self.num_robots = self.num_robots  
        self.agent_poses = None # robotarium convention poses
        self.prey_loc = None

        self.num_prey = 1
        self.num_agent = self.num_agent 
        self.terminated = False
        self.near_prey = [False]*self.num_agent # stores if agent_id has found the prey
        self.prey_captured = [False]*self.num_agent # stores if agent_id has captured the prey

        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}
        self.action_w2id = {v:k for k,v in self.action_id2w.items()}

        self.visualizer = Visualize( self.args ) # visualizer
        self.env = roboEnv(self, args) # robotarium Environment

        # Initialize the agents
        self.agents = []
        for agent_id in range(self.num_agent):
            self.agents.append( Agent(agent_id, args.predator_radius, args.predator_radius,\
                 self.action_id2w, self.action_w2id) ) 

        # Declaring action and observation space
        actions = []
        observations = []
        
        for agent in self.agents:
            actions.append(spaces.Discrete(5))
            obs_dim = 4
            observations.append(spaces.Box(low=-1.5, high=3, shape=(obs_dim,), dtype=np.float32))
        
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))

    
    def _generate_step_goal_positions(self, actions):
        
        '''
        Applies the actions on each agent.
        '''

        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:,i] = agent.generate_goal(goal[:,i], actions[i], self.args)
        
        return goal
    
    def _update_tracking_and_locations(self, agent_actions):
        
        '''
        Updates the environment's state. 
        '''

        prey_location = self.prey_loc
        
        # Iterate over all the agents and check if they are in the vicinity of the prey
        for i, agent in enumerate(self.agents):
            # Check if an agent is in the vicinity of the prey and if that is the case, 
            if not self.near_prey[i] and \
                np.linalg.norm(self.agent_poses[:2, agent.index] - prey_location) <= agent.sensing_radius:
                    self.near_prey[i] = True # agent can capture now 
                    # Now check the action for the agent and if it's no action, set the capture flag
                    if self.action_id2w[agent_actions[i]]:
                        self.prey_captured[i] = True

            # Check if prey has already been sensed and agent has not captured the prey
            elif self.near_prey[i] == True and self.prey_captured == False:
                # Now check the action for the agent and if it's no action, set the capture flag
                if self.action_id2w[agent_actions[i]]:
                    self.prey_captured[i] = True

        
    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
        x: Poses of all the robots
        '''
        state_space = {}
        state_space['poses'] = self.agent_poses
        state_space['prey'] = []
        state_space['prey'].append(np.array(self.prey_loc).reshape((2,1)))
        return state_space
    
    def reset(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.episode_steps = 0
        self.prey_locs = []
        self.num_prey = self.args.num_prey      
        
        width = self.args.ROBOT_INIT_RIGHT_THRESH - self.args.LEFT
        height = self.args.DOWN - self.args.UP
        self.agent_poses = generate_initial_locations(self.num_robots, width, height, self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.START_DIST)
        
        # Prey locations and tracking
        width = self.args.RIGHT - self.args.PREY_INIT_LEFT_THRESH
        self.prey_loc = generate_initial_locations(self.num_prey, width, height, self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.MIN_DIST, spawn_left=False)
        self.prey_loc = self.prey_loc[:2].T
        self.terminated = False
        self.prey_captured = [False] * self.num_prey
        self.prey_sensed = [False] * self.num_prey
        self.state_space = self._generate_state_space()
        self.env.reset()
        return [[0]*(4)] * self.num_robots
        
    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info (empty dictionary for now)
        '''
        self.episode_steps += 1
        
        # Steps into the environment and applies the action 
        # to get an updated state.
        self.env.step(actions_)
        self._update_tracking_and_locations(actions_)
        updated_state = self._generate_state_space()
        
        # get the observation and reward from the updated state
        obs     = self.get_observations(updated_state)
        rewards = self.get_rewards(updated_state)
        
        # condition for checking for the whether the episode is terminated
        if self.episode_steps > self.args.max_episode_steps:
            self.terminated = True             
        
        return obs, [rewards]*self.num_robots, [self.terminated]*self.num_robots, {} 

    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space

    def get_observations(self, state_space):
        
        observations = {}
        for agent in self.agents: 
            observations[agent.index] = agent.get_observation(state_space)    
        
        return observations[0]

    def get_rewards(self, state_space):
        '''

        '''
        agent_loc = state_space['poses']
        reward = 0
        reward += -(np.sum(np.square(agent_loc[:2].reshape(1,2) - self.prey_loc.reshape(1,2))))
        self.state_space = state_space
        return reward
    
    def render(self, mode='human'):
        # Render your environment
        pass

