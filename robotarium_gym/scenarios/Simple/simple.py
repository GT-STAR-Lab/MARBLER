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

class Agent:
    '''
    This is a helper class for Simple
    Keeps track of information for each agent and creates functions needed by each agent. 
    '''

    def __init__(self, index, action_id_to_word, action_word_to_id):
        self.index = index
        self.action_id2w = action_id_to_word
        self.action_w2id = action_word_to_id
        

    def get_observation( self, state_space):
        '''
        Returns: [agent_x_pos, agent_y_pos]
        array of dimension [1, OBS_DIM] 
        '''
        agent_pose = np.array(state_space['poses'][:, self.index ][:2])
        return agent_pose
    
    def generate_goal(self, goal_pose, action, args):
        '''
        Generates the final position for each time-step for the individual
        agent.
        '''

        if self.action_id2w[action] == 'left':
                goal_pose[0] = max( goal_pose[0] - args.step_dist, args.LEFT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
                goal_pose[0] = min( goal_pose[0] + args.step_dist, args.RIGHT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = max( goal_pose[1] - args.step_dist, args.UP)
        elif self.action_id2w[action] == 'down':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = min( goal_pose[1] + args.step_dist, args.DOWN)
        else:
             goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
             goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        
        return goal_pose

# An extremely simple environment for debugging the policy. 
# It consists of multiple agent who already knows the goal's location 
# and will get dense rewards. 

class simple(BaseEnv):
    def __init__(self, args):
        # Settings
        self.args = args

        self.agent_poses = None # robotarium convention poses
        self.goal_loc = None

        self.num_goal = 1 # There is only one goal
        self.num_agent = args.n_agents 
        self.num_robots = args.n_agents
        self.terminated = False

        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}
        self.action_w2id = {v:k for k,v in self.action_id2w.items()}
        
        self.visualizer = Visualize( self.args ) # visualizer
        self.env = roboEnv(self, args) # robotarium Environment

        # Initialize the agents
        self.agents = []
        for agent_id in range(self.num_agent):
            self.agents.append( Agent(agent_id, self.action_id2w, self.action_w2id) ) 

        # Declaring action and observation space
        actions = []
        observations = []
        
        for agent in self.agents:
            actions.append(spaces.Discrete(5))
            self.obs_dim = 2 * (self.num_agent + 1) # Total agents + goal locations
            observations.append(spaces.Box(low=-1.5, high=3, shape=(self.obs_dim,), dtype=np.float32))
        
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))
        self.adj_matrix = 1-np.identity(self.num_robots, dtype=int)

    
    def _generate_step_goal_positions(self, actions):
        
        '''
        Applies the actions on each agent.
        '''

        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:,i] = agent.generate_goal(goal[:,i], actions[i], self.args)
        
        return goal
        
    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
         - poses: Poses of all the robots
         - goal: Location of the goal 
        '''
        state_space = {}
        state_space['poses'] = self.agent_poses
        state_space['goal'] = []
        state_space['goal'].append(np.array(self.goal_loc).reshape((2,1)))
        return state_space
    
    def reset(self):
        '''
        Resets the environment before running the episode
        '''
        self.episode_steps = 0
        
        # Specify the area is which agent will be spawne
        width = self.args.ROBOT_INIT_RIGHT_THRESH - self.args.LEFT
        height = self.args.DOWN - self.args.UP
        # Agent pose 
        self.agent_poses = generate_initial_locations(self.num_robots, width, height,\
             self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.start_dist)
        
        # Goal location generation
        width = self.args.RIGHT - self.args.PREY_INIT_LEFT_THRESH
        self.goal_loc = generate_initial_locations(1, width, height,\
             self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.step_dist, spawn_left=False)
        self.goal_loc = self.goal_loc[:2].T
        
        # Reset episode flag
        self.terminated = False
        # Reset state space
        self.state_space = self._generate_state_space()
        self.env.reset()
        return [[0]*(self.obs_dim)] * self.num_agent
        
    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info (empty dictionary for now)
        '''
        self.episode_steps += 1
        
        # Steps into the environment and applies the action 
        # to get an updated state.
        return_msg = self.env.step(actions_)
        updated_state = self._generate_state_space()
        obs     = self.get_observations(updated_state)

        if return_msg == '':
            rewards = self.get_rewards(updated_state)
            self.terminated = self.episode_steps > self.args.max_episode_steps 
        else:
            print("Ending due to", return_msg)
            rewards = [-5]*self.num_robots
            self.terminated = True
                
        return obs, rewards, [self.terminated]*self.num_agent, {} 

    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space

    def get_observations(self, state_space):
        '''
        Return's the full observation for the agents.
        '''
        observations = []
        
        for agent in self.agents: 
            observations.append(agent.get_observation(state_space))    

        full_observations = []
        for i, agent in enumerate(self.agents):
            full_observations.append(observations[i])
            nbr_indices = [j for j in range(self.num_agent) if j != agent.index]
            for nbr_index in nbr_indices:
                full_observations[i] = np.concatenate( (full_observations[i],observations[nbr_index]) )
            
            goal_loc = state_space['goal'][0].reshape(-1)
            full_observations[i] = np.concatenate((full_observations[i], goal_loc))
         
        return full_observations

    def get_rewards(self, state_space):
        '''
        Returns dense rewards based on the negative of the distance between the current agent & goal
        '''
        agent_loc = state_space['poses']
        rewards = []

        for agent_id, agent in enumerate(self.agents):
            reward = -(np.sum\
                (np.square(agent_loc[:2, agent_id].reshape(1,2) - self.goal_loc.reshape(1,2))))
            reward *= self.args.reward_scaler
            rewards.append(reward)
        
        self.state_space = state_space
        return rewards
    
    def render(self, mode='human'):
        # Render your environment
        pass

