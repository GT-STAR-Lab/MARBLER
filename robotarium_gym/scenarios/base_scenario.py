
class BaseEnv(object):

    def _generate_state_space(self):
        raise NotImplementedError()

    def _update_tracking_and_locations(self):
        raise NotImplementedError()
    
    def step(self):
        raise NotImplementedError()
    
    def make_world(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def reward(self):
        raise NotImplementedError()
    
    def observation(self):
        raise NotImplementedError()
    