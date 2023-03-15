from gym.envs.registration import register

register(
  id="RoboPCPGrid-v0",                     # Environment ID.
  entry_point="robogym.pcpAgents:PCPWrapper",  # The entry point for the environment class
  kwargs={},
    )