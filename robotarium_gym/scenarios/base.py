#Every Robotarium Environment needs these methods to work
class BaseEnv(object):
    def get_action_space(self):
        raise NotImplementedError()

    def get_observation_space(self):
        raise NotImplementedError()
    
    def step(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
    
    def _generate_step_goal_positions():
        raise NotImplementedError()

class BaseVisualization():
    #How the markers get reset at the beginning of each episode
    def initialize_markers(self):
        raise NotImplementedError()
    
    #How the markers move at each robotarium step
    def update_markers(self):
        raise NotImplementedError()