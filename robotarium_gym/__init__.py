from gym.envs.registration import register

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