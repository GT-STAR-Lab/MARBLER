from rps.utilities.controllers import *
from rps.utilities.barrier_certificates import *

class Controller:
    def __init__(self):
        self.single_integrator_position_controller = create_si_position_controller()
        self.si_to_uni_dyn, self.uni_to_si_states = create_si_to_uni_mapping()
        self.si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    
    def set_velocities(self, agent_poses, goals):
        xi = self.uni_to_si_states(agent_poses)
        dxi = self.single_integrator_position_controller(xi, goals[:2][:])
        dxi = self.si_barrier_cert(dxi, xi)
        dxu = self.si_to_uni_dyn(dxi, agent_poses)
        return dxu