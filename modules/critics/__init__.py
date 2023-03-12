from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS
REGISTRY = {}

REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS


