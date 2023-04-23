import numpy as np
from gym import spaces
import copy
import yaml
import os

#This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from robotarium_gym.utilities.roboEnv import roboEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.PredatorCapturePrey.visualize import *
from robotarium_gym.scenarios.base import BaseScenario
from robotarium_gym.scenarios.PredatorCapturePrey.agent import Agent

class pcpAgents(BaseScenario):
    def __init__(self, args):
        # Settings
        self.args = args

        self.num_robots = args.predator + args.capture
        self.agent_poses = None # robotarium convention poses
        self.prey_loc = None

        self.num_prey = args.num_prey
        self.num_predators = args.predator
        self.num_capture = args.capture
        
        self._initialize_agents(args)
        self._initialize_actions_observations()

        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}
        self.action_w2id = {v:k for k,v in self.action_id2w.items()}

        self.visualizer = Visualize( self.args )
        self.env = roboEnv(self, args)
        
    def _initialize_agents(self, args):
        '''
        Initializes all agents and pushes them into a list - self.agents 
        predators first and then capture agents
        '''
        self.agents = []
        # Initialize predator agents
        for i in range(self.num_predators):
            self.agents.append( Agent(i, args.predator_radius, 0) )
        # Initialize capture agents
        for i in range(self.num_capture):
            self.agents.append( Agent(i + self.args.predator, 0, args.capture_radius) )

    def _initialize_actions_observations( self ):
        actions = []
        observations = []
        
        for agent in self.agents:
            actions.append(spaces.Discrete(5))
            ## This line seems too hacky. @Reza might want to look into it
            obs_dim = 6 * (self.args.num_neighbors + 1)
            observations.append(spaces.Box(low=-1.5, high=3, shape=(obs_dim,), dtype=np.float32))
        
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))

    def _generate_step_goal_positions(self, actions):
        '''
        User implemented
        Calculates the goal locations for the current agent poses and actions
        returns an array of the robotarium positions that it is trying to reach
        '''
        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:,i] = agent.generate_goal(goal[:,i], actions[i], self.args)
        
        return goal

    def _update_tracking_and_locations(self, agent_actions):
        # iterate over all the prey
        for i, prey_location in enumerate(self.prey_loc):
            # if the prey has already been captured, nothing to be done
            if self.prey_captured[i]:
                continue        
            #check if the prey has not been sensed
            if not self.prey_sensed[i]:
                # check if any of the agents has sensed it in the current step
                for agent in self.agents:
                    # check if any robot has it within its sensing radius
                    # print(self.agents.agent_poses[:2, agent.index], prey_location, np.linalg.norm(self.agents.agent_poses[:2, agent.index] - prey_location))
                    if np.linalg.norm(self.agent_poses[:2, agent.index] - prey_location) <= agent.sensing_radius:
                        self.prey_sensed[i] = True
                        break

            if self.prey_sensed[i]:
                # iterative over the agent_actions determined for each agent 
                for a, action in enumerate(agent_actions):
                    # check if any robot has no_action and has the prey within its capture radius if it is sensed already
                    if self.action_id2w[action]=='no_action'\
                        and np.linalg.norm(self.agent_poses[:2, self.agents[a].index] - prey_location) <= self.agents[a].capture_radius:
                        self.prey_captured[i] = True
                        break

    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
        x: Poses of all the robots
        '''
        state_space = {}
        state_space['poses'] = self.agent_poses
        state_space['num_prey'] = self.num_prey - sum(self.prey_captured) # number of prey not captured
        state_space['unseen_prey'] = self.num_prey - sum(self.prey_sensed) # number of prey unseen till now 
        state_space['prey'] = []

        # return locations of all prey that are not captured  till now
        for i in range(self.num_prey):
            if not self.prey_captured[i]:
                state_space['prey'].append(np.array(self.prey_loc[i]).reshape((2,1)))
        return state_space
    
    def reset(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.episode_steps = 0
        self.prey_locs = []
        self.num_prey = self.args.num_prey      
        
        # Agent locations
        width = self.args.ROBOT_INIT_RIGHT_THRESH - self.args.LEFT
        height = self.args.DOWN - self.args.UP
        self.agent_poses = generate_initial_locations(self.num_robots, width, height, self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.START_DIST)
        
        # Prey locations and tracking
        width = self.args.RIGHT - self.args.PREY_INIT_LEFT_THRESH
        self.prey_loc = generate_initial_locations(self.num_prey, width, height, self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.MIN_DIST, spawn_left=False)
        self.prey_loc = self.prey_loc[:2].T
        self.prey_captured = [False] * self.num_prey
        self.prey_sensed = [False] * self.num_prey
        
        self.state_space = self._generate_state_space()
        self.env.reset()
        # TODO: clean the empty observation returning
        return [[0]*(6 * (self.args.num_neighbors + 1))] * self.num_robots
        
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
        obs     = self.get_observations(updated_state)
        rewards = self.get_rewards(updated_state)
        
        # condition for checking for the whether the episode is terminated
        if self.episode_steps > self.args.max_episode_steps or \
            updated_state['num_prey'] == 0:
            terminated = True             
        
        return obs, [rewards]*self.num_robots, [terminated]*self.num_robots, {} 

    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space

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
            if self.args.delta > 0:
                nbr_indices = delta_disk_neighbors(state_space['poses'],agent.index,self.args.delta)
            elif self.args.num_neighbors >= self.num_robots-1:
                nbr_indices = [i for i in range(self.num_robots) if i != agent.index]
            else:
                nbr_indices = get_nearest_neighbors(state_space['poses'], agent.index, self.args.num_neighbors)
            
            # full_observation[i] is of dimension [NUM_NBRS, OBS_DIM]
            for nbr_index in nbr_indices:
                full_observations[i] = np.concatenate( (full_observations[i],observations[nbr_index]) )
        # dimension [NUM_AGENTS, NUM_NBRS, OBS_DIM]
        return full_observations

    def get_rewards(self, state_space):
        # Fully shared reward, this is a collaborative environment.
        reward = 0
        reward += (self.state_space['unseen_prey'] - state_space['unseen_prey']) * self.args.sense_reward
        reward += (self.state_space['num_prey'] - state_space['num_prey']) * self.args.capture_reward
        reward += self.args.time_penalty
        self.state_space = state_space
        return reward
    
    def render(self, mode='human'):
        # Render your environment
        pass
