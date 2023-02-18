import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import random
from pcpAgents import *

class PCPEnv:
    def __init__(self, pcpAgents, args):
        self.args = args
        self.agents = pcpAgents
        self.num_robots = self.args.sensing + self.args.capture
        self.first_run = True
        self.uni_barrier_cert = create_unicycle_barrier_certificate()

        if self.args.show_figure:
            self.sensing_marker_size_m = self.args.sensing_radius
            self.capture_marker_size_m = self.args.capture_radius
            self.goal_marker_size_m = 0.3
            self.line_width = 5
            self.CM = plt.cm.get_cmap('hsv', 4) # Agent/goal color scheme

    def run_episode(self):
        self._create_robotarium()

        state_space, x = self._generate_state_space() #x is the poses
        actions = self.agents.get_actions(state_space)
        while actions != []:
            #Set the velocities to each agent based on the assigned action
            velocities = self.uni_barrier_cert(actions, x) #makes sure no collisions
            self.robotarium.set_velocities(np.arange(self.num_robots), velocities)
            
            if self.args.show_figure:
                for i in range(x.shape[1]):
                    self.robot_markers[i].set_offsets(x[:2,i].T)

                    # Next two lines updates the marker sizes if the figure window size is changed. 
                    # They should be removed when submitting to the Robotarium.
                    self.robot_markers[i].set_sizes([determine_marker_size(self.robotarium, \
                                                        (self.sensing_marker_size_m if i < self.args.sensing else self.capture_marker_size_m))])
                self.goal_marker.set_sizes([determine_marker_size(self.robotarium, self.goal_marker_size_m)])

            self.robotarium.step()
            state_space, x = self._generate_state_space()
            actions = self.agents.get_actions(state_space)

    def _create_robotarium(self):
        if self.first_run:
            self.first_run = False
        else:
            self.robotarium.call_at_scripts_end() #TODO: check if this is needed and how it affects runtime
        
        #generate initial robot locations
        #Assumes y and theta can be anything but the x locations are within the left third of the robotarium
        initial_conditions = generate_initial_conditions(self.num_robots, width=1)
        for i in range(len(initial_conditions[0])):
            initial_conditions[0][i] -= 1

        self.robotarium = robotarium.Robotarium(number_of_robots= self.num_robots, show_figure = self.args.show_figure, \
                                                initial_conditions=initial_conditions, sim_in_real_time=self.args.real_time)
        
        #setting the prey location to a random location where:
        #   x is [0,1.4]
        #   y is [-.9, .9]
        prey_x = random.random() * 1.4
        prey_y = random.random() * 1.8 - .9
        self.prey_loc = [prey_x, prey_y]

        if self.args.show_figure:
            x = self.robotarium.get_poses()

            marker_size_sensing = determine_marker_size(self.robotarium, self.sensing_marker_size_m)
            marker_size_capture = determine_marker_size(self.robotarium, self.capture_marker_size_m)
            marker_size_goal = determine_marker_size(self.robotarium,self.goal_marker_size_m)
            self.robot_markers = [self.robotarium.axes.scatter( \
                x[0,ii], x[1,ii], s=(marker_size_sensing if ii < self.args.sensing else marker_size_capture), marker='o', facecolors='none',edgecolors=self.CM(0 if ii < self.args.sensing else 1),linewidth=self.line_width) 
                for ii in range(self.num_robots)]
            
            self.goal_marker = self.robotarium.axes.scatter( \
                prey_x, prey_y, s=marker_size_goal, marker='o', facecolors='none',edgecolors=self.CM(2),linewidth=self.line_width,zorder=-2)

        self.robotarium.step()

    def _generate_state_space(self):
        state_space = {}
        x = self.robotarium.get_poses()
        state_space['poses'] = x
        state_space['prey'] = self.prey_loc
        
        return state_space, x


    def __del__(self):
        self.robotarium.call_at_scripts_end()

