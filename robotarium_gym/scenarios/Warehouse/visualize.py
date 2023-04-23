from rps.utilities.misc import *
import matplotlib.patches as patches

from robotarium_gym.scenarios.base import BaseVisualization

class Visualize(BaseVisualization):
    def __init__(self,args):
        self.args = args
        self.agent_marker_size_m = .2
        self.line_width = 1
        self.CM = plt.cm.get_cmap('hsv', 10) # Agent/goal color scheme
        self.show_figure = True

    def initialize_markers(self, robotarium, agents):
        agent_marker_size = determine_marker_size(robotarium, self.agent_marker_size_m)

        self.goals = []
        w = self.args.goal_width
        self.goals.append(robotarium.axes.add_patch(patches.Rectangle([-1.5,-1], w,1, color=self.CM(1), zorder=-1)))
        self.goals.append(robotarium.axes.add_patch(patches.Rectangle([-1.5,0], w,1, color=self.CM(2), zorder=-1)))
        self.goals.append(robotarium.axes.add_patch(patches.Rectangle([1.5-w,-1], w,1, color=self.CM(2), zorder=-1)))
        self.goals.append(robotarium.axes.add_patch(patches.Rectangle([1.5-w,0], w,1, color=self.CM(1), zorder=-1)))

        self.robot_markers = [ robotarium.axes.scatter( \
                agents.agent_poses[0,ii], agents.agent_poses[1,ii],
                s=agent_marker_size, marker='o', facecolors='none',\
                edgecolors = (self.CM(3) if ii%2 == 0 else self.CM(0)), linewidth=self.line_width )\
                for ii in range(agents.num_robots) ]

    def update_markers(self, robotarium, agents):
        for i in range(agents.agent_poses.shape[1]):
            self.robot_markers[i].set_offsets(agents.agent_poses[:2,i].T)
            # Next two lines updates the marker sizes if the figure window size is changed.
            self.robot_markers[i].set_sizes([determine_marker_size(robotarium, self.agent_marker_size_m)])
