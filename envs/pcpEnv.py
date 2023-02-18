import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import random
from pcpAgents import *

class PCPEnv:
    def __init__(self, pcpAgents, args):
        self.args = args
        self.agents = pcpAgents
        self.num_robots = self.args.sensing + self.args.capture
        self.first_run = True

    def run_episode(self):
        self._create_robotarium()

        actions = self.agents.get_actions(self._generate_state_space())
        while actions != []:
            #Set the velocities to each agent based on the assigned action
            
            if self.args.show_figure:
                x = self.robotarium.get_poses()
                for i in range(x.shape[1]):
                    self.robot_markers[i].set_offsets(x[:2,i].T)

                    # Next two lines updates the marker sizes if the figure window size is changed. 
                    # They should be removed when submitting to the Robotarium.
                    self.robot_markers[i].set_sizes([determine_marker_size(self.robotarium, self.robot_marker_size_m)])
                self.goal_marker.set_sizes([determine_marker_size(self.robotarium, self.goal_marker_size_m)])

            self.robotarium.step()
            actions = self.agents.get_actions(self._generate_state_space())

    def _create_robotarium(self):
        if self.first_run:
            self.first_run = False
        else:
            self.robotarium.call_at_scripts_end() #TODO: check if this is needed and how it affects runtime
        
        #generate initial locations
        #Assumes y and theta can be anything but the x locations are within the left third of the robotarium
        initial_conditions = generate_initial_conditions(self.num_robots, width=1)
        for i in range(len(initial_conditions[0])):
            initial_conditions[0][i] -= 1

        self.robotarium = robotarium.Robotarium(number_of_robots= self.num_robots, show_figure = self.args.show_figure, \
                                                initial_conditions=initial_conditions, sim_in_real_time=self.args.real_time)
        
        #setting the prey location to a random location where:
        #   x is [0,1.5]
        #   y is [-.9, .9]
        prey_x = random.random() * 1.5
        prey_y = random.random() * 1.8 - .9
        self.prey_loc = [prey_x, prey_y]

        if self.args.show_figure:
            x = self.robotarium.get_poses()
            self.robot_marker_size_m = 0.2
            self.goal_marker_size_m = 0.3
            line_width = 5
            CM = plt.cm.get_cmap('hsv', 3) # Agent/goal color scheme

            marker_size_robot = determine_marker_size(self.robotarium, self.robot_marker_size_m)
            marker_size_goal = determine_marker_size(self.robotarium,self.goal_marker_size_m)
            self.robot_markers = [self.robotarium.axes.scatter( \
                x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors=CM(0 if ii < self.args.sensing else 1),linewidth=line_width) 
                for ii in range(self.num_robots)]
            
            self.goal_marker = self.robotarium.axes.scatter( \
                prey_x, prey_y, s=marker_size_goal, marker='o', facecolors='none',edgecolors=CM(2),linewidth=line_width,zorder=-2)

        self.robotarium.step()

    def _generate_state_space(self):
        pass

    def __del__(self):
        self.robotarium.call_at_scripts_end()

