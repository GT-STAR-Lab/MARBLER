#Every Robotarium Environment needs these methods to work
class BaseEnv(object):
    def get_action_space(self):
        #Must return the action space
        raise NotImplementedError()

    def get_observation_space(self):
        #Must return the observation space
        raise NotImplementedError()
    
    def step(self):
        #Must return [observations, rewards, done, info]
        raise NotImplementedError()

    def reset(self):
        #Must return an observation array
        raise NotImplementedError()
    
    def render(self, mode='human'):
        #This isn't really used in our environments
        pass
    
    def _generate_step_goal_positions():
        #Must return goal locations for each agent
        raise NotImplementedError()

class BaseVisualization():
    #How the markers get reset at the beginning of each episode
    def initialize_markers(self):
        raise NotImplementedError()
    
    #How the markers move at each robotarium step
    def update_markers(self):
        raise NotImplementedError()