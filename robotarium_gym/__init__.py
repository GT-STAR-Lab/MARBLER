from gym.envs.registration import register
#import robotarium_gym.scenarios as scenarios
# Multiagent envs
# ----------------------------------------

_particles = {
    "pcp": "PredatorCapturePrey-v0",
}

for scenario_name, gymkey in _particles.items():
    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point=f"robotarium_gym.scenarios.{scenario_name}.wrapper:Wrapper",
        kwargs={},
    )

'''
from gym.envs.registration import register

register(
  id="RoboPCPGrid-v0",                     # Environment ID.
  entry_point="scenarios.pcpAgents:PCPWrapper",  # The entry point for the environment class
  kwargs={},
    )
'''