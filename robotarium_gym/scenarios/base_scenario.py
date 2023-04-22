
class BaseEnv(object):

    def _generate_state_space(self):
        raise NotImplementedError()
    
    def reset():
        raise NotImplementedError()
    
    def step():
        raise NotImplementedError()
    
    def get_rewards():
        raise NotImplementedError()