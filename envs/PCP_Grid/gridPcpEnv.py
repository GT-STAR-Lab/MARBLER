import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import random
import time

from PCP_Grid import gridPcpAgents

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
        self.num_robots = self.args.predator + self.args.capture
        self.first_run = True 

        self.single_integrator_position_controller = create_si_position_controller()
        _, self.uni_to_si_states = create_si_to_uni_mapping()
        self.si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()
        self.si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

        self.width = self.args.grid_size * 3
        self.height = self.args.grid_size * 2
        self.grid_locations = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(self._get_real_width_height(j, i))
            self.grid_locations.append(row)
        
        #Will spawn agents in left third of the robotarium
        if self.num_robots > 2 * self.args.grid_size ** 2:
            print('Error, too many agents for the grid size')
            exit(-1)
        self.agent_poses = [[0,0]] * self.num_robots

        if self.args.show_figure:
            self.predator_marker_size_m = (self.args.predator_radius - .5) / self.args.grid_size 
            self.capture_marker_size_m = (.05 if self.args.capture_radius == 0 else (self.args.capture_radius - .5) / self.args.grid_size)
            self.goal_marker_size_m = .05
            self.line_width = 5
            self.CM = plt.cm.get_cmap('hsv', 4) # Agent/goal color scheme


    def run_episode(self):
        '''
        Creates a new instance of the robotarium and runs the agents until PCPAgents.get_actions returns []
        '''
        self._create_robotarium()

        state_space, x = self._generate_state_space() #x is the poses. Can only get the poses once per step
        actions = self.agents.get_actions(state_space)

        iterations = 0
        while actions != []:
            if iterations % self.args.update_frequency == 0:
                self._update_poses(actions)
                
            #uses the robotarium commands to get the velocities of each robot
            x_si = self.uni_to_si_states(x)
            goals = self._generate_goal_positions()
            dxi = self.single_integrator_position_controller(x_si, goals[:2][:])
            dxi = self.si_barrier_cert(dxi, x_si)
            dxu = self.si_to_uni_dyn(dxi, x)
            self.robotarium.set_velocities(np.arange(self.num_robots), dxu)
            
            if self.args.show_figure:
                for i in range(x.shape[1]):
                    self.robot_markers[i].set_offsets(x[:2,i].T)

                    # Next two lines updates the marker sizes if the figure window size is changed. 
                    # They should be removed when submitting to the Robotarium.
                    self.robot_markers[i].set_sizes([determine_marker_size(self.robotarium, \
                                                        (self.predator_marker_size_m if i < self.args.predator else self.capture_marker_size_m))])
                self.goal_marker.set_sizes([determine_marker_size(self.robotarium, self.goal_marker_size_m)])

            self.robotarium.step()
            iterations += 1
            state_space, x = self._generate_state_space()
            if iterations % self.args.update_frequency == 0:
                actions = self.agents.get_actions(state_space)           
            
            
    def _update_poses(self, actions):
        for i in range(self.num_robots):
            match actions[i]['Velocity']:
                case 'left':
                    self.agent_poses[i] = [max(self.agent_poses[i][0]-1, 0), self.agent_poses[i][1]]
                case 'right':
                    self.agent_poses[i] = [min(self.agent_poses[i][0]+1, self.width-1), self.agent_poses[i][1]]
                case 'up':
                    self.agent_poses[i] = [self.agent_poses[i][0], max(self.agent_poses[i][1]-1, 0)]
                case 'down':
                    self.agent_poses[i] = [self.agent_poses[i][0], min(self.agent_poses[i][1]+1, self.height-1)]
                case _:
                    continue #if 'stop' or 'capture' the agent i's pose does not change

    def _generate_goal_positions(self):
        '''
        Using the positions from self.agent_poses as the robot goal locations on the grid,
        returns an array of the robotarium positions that it is trying to reach
        '''
        goal_z = [0] * self.num_robots
        goal_x = []
        goal_y = []
        for i in self.agent_poses:
            pose = self._get_real_width_height(i[0], i[1])
            goal_x.append(pose[0])
            goal_y.append(pose[1])

        return np.array([goal_x, goal_y, goal_z])

    def _get_real_width_height(self, x, y):
        '''
        Given a grid coordinate, returns the coordinate in the robotarium units
        '''
        x = (3 * x / self.width - 1.5) + (3 / self.width / 2)
        y = (2 * y / self.height - 1) + (2 / self.height / 2)
        return [x,y]
    
    def _create_robotarium(self):
        '''
        Creates a new instance of the robotarium
        Randomly initializes the prey in the right half and the agents in the left third of the Robotarium
        '''
        if self.first_run:
            self.first_run = False
        else:
            self.robotarium.call_at_scripts_end() #TODO: check if this is needed and how it affects runtime
        
        #generate initial robot locations
        #Assumes y can be anything but the x locations are within the left third of the robotarium
        initial_values = np.random.choice(2 * self.args.grid_size ** 2, self.num_robots, replace = False)
        for i in range(len(initial_values)):
            self.agent_poses[i] = [initial_values[i]%self.args.grid_size, int(initial_values[i]/self.args.grid_size)]

        initial_conditions = self._generate_goal_positions()

        self.robotarium = robotarium.Robotarium(number_of_robots= self.num_robots, show_figure = self.args.show_figure, \
                                                initial_conditions=initial_conditions, sim_in_real_time=self.args.real_time)
        
        #setting the prey location to a random location in the right third of the grid
        prey_loc = np.random.choice(2 * self.args.grid_size ** 2, 1)
        self.prey_loc = self._get_real_width_height(prey_loc[0]%self.args.grid_size + self.width*(2/3), int(prey_loc[0] / self.args.grid_size))

        if self.args.show_figure:
            x = self.robotarium.get_poses()

            marker_size_predator = determine_marker_size(self.robotarium, self.predator_marker_size_m)
            marker_size_capture = determine_marker_size(self.robotarium, self.capture_marker_size_m)
            marker_size_goal = determine_marker_size(self.robotarium,self.goal_marker_size_m)
            self.robot_markers = [self.robotarium.axes.scatter( \
                x[0,ii], x[1,ii], s=(marker_size_predator if ii < self.args.predator else marker_size_capture), marker='o', facecolors='none',edgecolors=self.CM(0 if ii < self.args.predator else 1),linewidth=self.line_width) 
                for ii in range(self.num_robots)]
            
            self.goal_marker = self.robotarium.axes.scatter( self.prey_loc[0], self.prey_loc[1], \
                s=marker_size_goal, marker='.', facecolors='none',edgecolors=self.CM(2),linewidth=self.line_width,zorder=-2)

        self.robotarium.step()

    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
        '''
        state_space = {}
        x = self.robotarium.get_poses()
        state_space['poses'] = x
        state_space['prey'] = np.array(self.prey_loc).reshape((2,1))
        
        return state_space, x


    def __del__(self):
        self.robotarium.call_at_scripts_end()

