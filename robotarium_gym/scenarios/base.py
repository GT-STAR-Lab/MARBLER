
class BaseScenario(object):

    def get_action_space(self):
        raise NotImplementedError()

    def get_observation_space(self):
        raise NotImplementedError()
    
    def step(self):
        raise NotImplementedError()

    def reset(self, world):
        raise NotImplementedError()
    
class BaseVisualization():
    def initial_markers(self):
        raise NotImplementedError()
    
    def update_markers(self):
        raise NotImplementedError()