from gym import spaces
import numpy as np
import copy
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.MaterialTransport.visualize import Visualize
from robotarium_gym.utilities.roboEnv import roboEnv


class Agent:
    #These agents are specifically implimented for the warehouse scenario
    def __init__(self, index, action_id_to_word, action_word_to_id, torque, speed):
        self.index = index
        self.torque = torque
        self.speed = speed
        self.load = 0
        self.action_id2w = action_id_to_word
        self.action_w2id = action_word_to_id
   
    def generate_goal(self, goal_pose, action, args):    
        '''
        updates the goal_pose based on the agent's actions
        '''   
        action = action // 4 #This is to account for the messages
        if self.action_id2w[action] == 'left':
                goal_pose[0] = max( goal_pose[0] - self.speed, args.LEFT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
                goal_pose[0] = min( goal_pose[0] + self.speed, args.RIGHT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = max( goal_pose[1] - self.speed, args.UP)
        elif self.action_id2w[action] == 'down':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = min( goal_pose[1] + self.speed, args.DOWN)
        else:
             goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
             goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        
        return goal_pose

class MaterialTransport(BaseEnv):
    def __init__(self, args):
        self.args = args
        self.num_robots = self.args.n_agents
        self.agent_poses = None

        #Agent agent's observation is [pos_x,pos_y,load, zone1_load, zone2_load, a1_message, a2_message, a3_message, a4_message, speed, torque] 
        #   where speed and torque are only included if capability_aware is true
        if self.args.capability_aware:
            self.agent_obs_dim = 11
        else:
            self.agent_obs_dim = 9

        self.zone1_args = copy.deepcopy(self.args.zone1)
        del self.zone1_args['distribution']   
        self.zone2_args = copy.deepcopy(self.args.zone2)
        del self.zone2_args['distribution']  
        
        #This isn't really needed but makes a bunch of stuff clearer
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}
        self.action_w2id = {v:k for k,v in self.action_id2w.items()}
        
        self.agents = []
        for i in range(self.args.n_fast_agents):
            self.agents.append(Agent(i, self.action_id2w, self.action_w2id, self.args.small_torque, self.args.fast_step))
        for i in range(self.args.n_fast_agents, self.args.n_fast_agents+self.args.n_slow_agents):
            self.agents.append(Agent(i, self.action_id2w, self.action_w2id, self.args.large_torque, self.args.slow_step))

        #Initializes the action and observation spaces
        actions = []
        observations = []
        for a in self.agents:
            actions.append(spaces.Discrete(20))
            #each agent's observation is a tuple of size 3
            #the minimum observation is the left corner of the robotarium, the maximum is the righ corner
            observations.append(spaces.Box(low=-1.5, high=1.5, shape=(self.agent_obs_dim,), dtype=np.float32))
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))
        
        self.visualizer = Visualize(self.args) #needed for Robotarium renderings
        #Env impliments the robotarium backend
        #It expects to have access to agent_poses, visualizer, num_robots and _generate_step_goal_positions
        self.env = roboEnv(self, args)  

    def reset(self):
        self.episode_steps = 0
        self.messages = [0,0,0,0] 

        #Randomly sets the load for each zone
        self.zone1_load = int(getattr(np.random, self.args.zone1['distribution'])(**self.zone1_args))
        self.zone2_load = int(getattr(np.random, self.args.zone2['distribution'])(**self.zone2_args))
        
        for a in self.agents:
            a.load=0
        
        #Generate the agent locations based on the config
        width = self.args.end_goal_width
        height = self.args.DOWN - self.args.UP
        #Agents can spawn in the Robotarium between UP, DOWN, LEFT and LEFT+end_goal_width for this scenario
        self.agent_poses = generate_initial_locations(self.num_robots, width, height, self.args.LEFT+self.args.end_goal_width, start_dist=self.args.start_dist)
        self.env.reset()
        return [[0]*self.agent_obs_dim] * self.num_robots
    
    def step(self, actions_):
        self.episode_steps += 1

        #Robotarium actions and updating agent_poses all happen here
        return_message = self.env.step(actions_)
        for i in range(len(self.messages)):
            self.messages[i] = actions_[i] % 4

        obs = self.get_observations()
        if return_message == '':
            reward = self.get_reward()       
            terminated = self.episode_steps > self.args.max_episode_steps #For this environment, episode only ends after timing out
            #Terminates when all agent loads are 0 and the goal zone loads are 0
            if not terminated:
                terminated = self.zone1_load == 0 and self.zone2_load == 0
                if terminated:
                    for a in self.agents:
                        if a.load != 0:
                            terminated = False
                            break
        else:
            print("Ending due to", return_message)
            reward = -6
            terminated = True
        return obs, [reward] * self.num_robots, [terminated]*self.num_robots, {}
    
    def get_observations(self):
        observations = [] #Each agent's individual observation
        for a in self.agents:
            if self.args.capability_aware:
                observations.append([*self.agent_poses[:, a.index ][:2], a.load, \
                                     self.zone1_load, self.zone2_load, *self.messages, a.torque, a.speed])
            else:
                observations.append([*self.agent_poses[:, a.index ][:2], a.load, \
                                     self.zone1_load, self.zone2_load, *self.messages])
        return observations

    def get_reward(self):
        '''
        Agents take a small penalty every step and get a reward when they unload proportional to how much load they are carrying
        '''
        reward = self.args.time_penalty
        for a in self.agents:
            pos = self.agent_poses[:, a.index ][:2]
            if a.load > 0:             
                if pos[0] < -1.5 + self.args.end_goal_width:
                    reward += a.load * self.args.unload_multiplier
                    a.load = 0
            else:
                if pos[0] > 1.5 - self.args.end_goal_width:
                    if self.zone2_load > a.torque:              
                        a.load = a.torque
                        self.zone2_load -= a.torque
                    else:
                        a.load = self.zone2_load
                        self.zone2_load = 0
                    reward += a.load * self.args.load_multiplier
                elif np.linalg.norm(self.agent_poses[:2, a.index] - [0, 0]) <= self.args.zone1_radius:
                    if self.zone1_load > a.torque:              
                        a.load = a.torque
                        self.zone1_load -= a.torque
                    else:
                        a.load = self.zone1_load
                        self.zone1_load = 0
                    reward += a.load * self.args.load_multiplier
        return reward

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

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space