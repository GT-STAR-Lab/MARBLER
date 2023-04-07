import copy
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import random
import time
from gym import spaces

from pcpAgents import *
from utilities import *

class PCPEnv:
    def __init__(self, pcpAgents, args):
        '''
        Inputs: object of the PCPAgents class and an argparse object
        Argparse object must at least contain:
            Number of predator robots as args.predator
            Number of capture robots as args.capture
            Radii for the predator and capture agents as args.predator_radius and args.capture radius respectively
            Whether or not to show the figure as args.show_figure
            Whether or not to run the simulation in real time as args.real_time
        '''
        self.args = args
        self.agents = pcpAgents
        self.num_prey = self.args.num_prey
        self.prey_captured = [False] * self.num_prey
        self.prey_sensed = [False] * self.num_prey
        self.num_robots = self.args.predator + self.args.capture
        self.first_run = True 
        self.episodes = 0

        self.single_integrator_position_controller = create_si_position_controller()
        self.si_to_uni_dyn, self.uni_to_si_states = create_si_to_uni_mapping()
        self.si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

        self.agent_poses = None

        self.predator_marker_size_m = self.args.predator_radius
        self.capture_marker_size_m = self.args.capture_radius
        self.goal_marker_size_m = .05
        self.line_width = 5
        self.CM = plt.cm.get_cmap('hsv', 5) # Agent/goal color scheme
        
        # define the observation space & action space for the agents
        self.action_space = []
        self.observation_space = []
        
        for agent in self.agents.agents:
            self.action_space.append(spaces.Discrete(5))
            obs_dim = 6 * (self.args.num_neighbors + 1)
            self.observation_space.append(spaces.Box(low=-1.5, high=3, shape=(obs_dim,), dtype=np.float32))
        
        self.action_space = spaces.Tuple(tuple(self.action_space))
        self.observation_space = spaces.Tuple(tuple(self.observation_space))

        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}
        self.action_w2id = {v:k for k,v in self.action_id2w.items()}

    def reset(self):
        '''
        Reset the environment
        '''
        self.num_prey = self.args.num_prey
        self.prey_captured = [False] * self.num_prey
        self.prey_sensed = [False] * self.num_prey
        self.prey_left = [i for i in range(self.num_prey)]
        self.episodes += 1
        self._create_robotarium()

    def step(self, actions_):
        '''
        Take a step into the environment given the action
        '''

        # Considering one step to be equivalent to 60 iters
        for iterations in range(self.args.update_frequency):
            
            # Get the actual position of the agents
            self.agent_poses = self.robotarium.get_poses()

            if iterations == 0:
                goals_ = self._generate_goal_positions(actions_)

            # Uses the robotarium commands to get the velocities of each robot   
            # Only does this once every 10 steps because otherwise training is really slow 
            if iterations % 10 == 0 or self.args.robotarium:                    
                xi = self.uni_to_si_states(self.agent_poses)
                dxi = self.single_integrator_position_controller(xi, goals_[:2][:])
                dxi = self.si_barrier_cert(dxi, xi)
                dxu = self.si_to_uni_dyn(dxi, self.agent_poses)
                self.robotarium.set_velocities(np.arange(self.num_robots), dxu)
            
            if self.show_figure:
                for i in range(self.agent_poses.shape[1]):
                    self.robot_markers[i].set_offsets(self.agent_poses[:2,i].T)

                    # Next two lines updates the marker sizes if the figure window size is changed. 
                    # They should be removed when submitting to the Robotarium.
                    self.robot_markers[i].set_sizes([determine_marker_size(self.robotarium, \
                                                        (self.predator_marker_size_m if i < self.args.predator else self.capture_marker_size_m))])
                #self.goal_marker.set_sizes([determine_marker_size(self.robotarium, self.goal_marker_size_m)])
                for i in range(len(self.prey_markers)):
                    if not self.prey_captured[i]:
                        self.prey_markers[i].set_sizes([determine_marker_size(self.robotarium, self.goal_marker_size_m)])
                        if self.prey_sensed[i]:
                            self.prey_markers[i].set_facecolor(self.CM(4))
                    else:
                        self.prey_markers[i].set_sizes([0,0])

            self.robotarium.step()
        
        self._update_prey_status(actions_)
        return self._generate_state_space()

    def _generate_goal_positions(self, actions):
        '''
        Using the positions from self.agent_poses as the robot goal locations on the grid,
        returns an array of the robotarium positions that it is trying to reach
        '''
        goal = copy.deepcopy(self.agent_poses)
        for i in range(self.num_robots):
            if self.action_id2w[actions[i]] == 'left':
                    goal[0][i] = max( goal[0][i] - self.args.MIN_DIST, self.args.LEFT)
            elif self.action_id2w[actions[i]] == 'right':
                    goal[0][i] = min( goal[0][i] + self.args.MIN_DIST, self.args.RIGHT)
            elif self.action_id2w[actions[i]] == 'up':
                    goal[1][i] = max( goal[1][i] - self.args.MIN_DIST, self.args.UP)
            elif self.action_id2w[actions[i]] == 'down':
                    goal[1][i] = min( goal[1][i] + self.args.MIN_DIST, self.args.DOWN)
            else:
                    continue #if 'stop' or 'capture' the agent i's pose does not change
        return goal
    
    def _create_robotarium(self):
        '''
        Creates a new instance of the robotarium
        Randomly initializes the prey in the right half and the agents in the left third of the Robotarium
        '''
        if self.first_run:
            self.first_run = False
        else:
            #self.robotarium.call_at_scripts_end() #TODO: check if this is needed and how it affects runtime
            del self.robotarium

        #generate initial robot locations
        #Assumes y can be anything but the x locations are within the left third of the robotarium
        initial_conditions = self._generate_locations( self.num_robots, right = self.args.ROBOT_INIT_RIGHT_THRESH)
        # Robotarium inbuilt code -
        # initial_conditions = generate_initial_conditions( self.num_robots, spacing=0.2, width=2.8, height=1.8 )

        # Figure showing and setting up robotarium
        if self.args.show_figure_frequency != -1 and self.episodes % self.args.show_figure_frequency == 0:
            self.show_figure = True
        else: 
            self.show_figure = False
        self.robotarium = robotarium.Robotarium(number_of_robots= self.num_robots, show_figure = self.show_figure, \
                                                initial_conditions=initial_conditions, sim_in_real_time=self.args.real_time)
        
        self.prey_loc = self._generate_locations(self.num_prey, left = self.args.PREY_INIT_LEFT_THRESH, robotarium_poses = False)

        self.agent_poses = self.robotarium.get_poses()
        if self.show_figure:
            marker_size_predator = determine_marker_size(self.robotarium, self.predator_marker_size_m)
            marker_size_capture = determine_marker_size(self.robotarium, self.capture_marker_size_m)
            marker_size_goal = determine_marker_size(self.robotarium,self.goal_marker_size_m)            

            self.robot_markers = [self.robotarium.axes.scatter( \
                    self.agent_poses[0,ii], self.agent_poses[1,ii], s=(marker_size_predator if ii < self.args.predator else marker_size_capture), \
                    marker='o', facecolors='none', edgecolors = self.CM(0 if ii < self.args.predator else 1), linewidth=self.line_width )\
                    for ii in range(self.num_robots)]
            
            self.prey_markers = [self.robotarium.axes.scatter( \
                    self.prey_loc[ii][0], self.prey_loc[ii][1], s=marker_size_goal, marker='.', facecolors=self.CM(2), edgecolors=self.CM(2),\
                    linewidth=self.line_width, zorder=-2) for ii in range(self.num_prey)]

        self.robotarium.step()

    def _generate_locations(self, num_robots, left = None, right = None,\
                        robotarium_poses = True, min_dist = None ):
        if left  == None: left  = self.args.LEFT 
        if right == None: right = self.args.RIGHT
        up   = self.args.UP
        down = self.args.DOWN
        if min_dist == None: min_dist = self.args.MIN_DIST
        buffer = self.args.MIN_DIST

        # overlay a grid over the allowed region
        cols = int( round( (right - left )/min_dist, 0)) - 1
        rows = int( round( (down - up )/min_dist, 0)) - 1

        # pick random locations from the grid
        grid_indices = np.random.choice( rows*cols, num_robots, replace = False)
        # convert grid locations back to robotarium coordinates
        locations = []
        for loc in grid_indices:
            locations.append([left + buffer + ( (loc % cols) * min_dist ),\
                                up + buffer + ( int(loc / cols) * min_dist ) ])

        if robotarium_poses:
            return convert_to_robotarium_poses(locations)

        return locations

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

    def _update_prey_status(self, actions):
        agents = self.agents.agents
        # removeIndices = []
        # iterate over all the prey
        for i, prey_location in enumerate(self.prey_loc):
            # if the prey has already been captured, nothing to be done
            if self.prey_captured[i]:
                continue
            
            #check if the prey has not been sensed
            if not self.prey_sensed[i]:
                # check if any of the agents has sensed it in the current step
                for agent in agents:
                    # check if any robot has it within its sensing radius
                    print(self.agent_poses[:2, agent.index], prey_location, np.linalg.norm(self.agent_poses[:2, agent.index] - prey_location))
                    if np.linalg.norm(self.agent_poses[:2, agent.index] - prey_location) <= agent.sensing_radius:
                        self.prey_sensed[i] = True
                        print("prey id captured", i, time.time())
                        break

            if self.prey_sensed[i]:
                # iterative over the actions determined for each agent 
                for a, action in enumerate(actions):
                    # check if any robot has no_action and has the prey within its capture radius if it is sensed already
                    if self.action_id2w[action]=='no_action'\
                        and np.linalg.norm(self.agent_poses[:2, agents[a].index] - prey_location) <= agents[a].capture_radius:
                        self.prey_captured[i] = True
                        break

    def __del__(self):
        self.robotarium.call_at_scripts_end()

