from rps.utilities.controllers import *
from rps.utilities.barrier_certificates import *

class Controller:
    def __init__(self, type='safe', custom=None):
        '''
        Types are: "safe", "default", or "custom"
        If type is set to custom, a custom controller much be given
        If type is set to "default", there is a high probability of collisions when submitting to the Robotarium
        '''
        self.single_integrator_position_controller = create_si_position_controller()
        self.si_to_uni_dyn, self.uni_to_si_states = create_si_to_uni_mapping()
        if type == "safe":
            self.si_barrier_cert = create_single_integrator_barrier_certificate2(safety_radius=.2)
        elif type == "default":
            self.si_barrier_cert = create_single_integrator_barrier_certificate()
        else:
            self.si_barrier_cert = custom
    
    def set_velocities(self, agent_poses, goals):
        xi = self.uni_to_si_states(agent_poses)
        dxi = self.single_integrator_position_controller(xi, goals[:2][:])
        dxi = self.si_barrier_cert(dxi, xi)
        dxu = self.si_to_uni_dyn(dxi, agent_poses)
        return dxu