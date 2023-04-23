class World:
    def __init__(self):
        left=-1.5
        right = -1.5
        up = -.9
        down = .9
        step_dist = .2
        start_dist = .3
        robotarium = False
        real_time = False
        robotarium_update_frequency = 33 #33 steps per second
        show_figure_frequency = -1


class Agent:
    def __init__(self):
        self.color = None
        self.radius = None
        self.pos = None
        self.discrete = True

class Landmark:
    def __init__(self):
        self.color = None
        self.radius = None
        self.pos = None
        self.radius = None