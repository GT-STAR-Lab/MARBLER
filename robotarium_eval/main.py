import torch
import json
import importlib
import numpy as np
import yaml

config_path = "config.npy"

class DictView(object):
        def __init__(self, d):
            self.__dict__ = d

def load_model(args):
    model_config = open(args.model_config_file)
    model_config = DictView(json.load(model_config))
    model_config.n_actions = args.n_actions

    params = torch.load(args.model_file, map_location=torch.device('cpu'))
    input_dim = params[list(params.keys())[0]].shape[1]
    print(input_dim)

    actor = importlib.import_module(args.actor_file)
    actor = getattr(actor, args.actor_class)
    
    model = actor(input_dim, model_config)
    model.load_state_dict(params)

    print(model.eval())
    return model, model_config

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

args = DictView(config)
model, model_config = load_model(args)

env_module = importlib.import_module(args.env_file)
env_class = getattr(env_module, args.env_class)
env = env_class(args)

obs = np.array(env.reset())
n_agents = len(obs)
hs = [np.zeros((model_config.hidden_dim, )) for i in range(n_agents)]

for i in range(args.steps):
    if model_config.obs_agent_id:
        obs = np.concatenate([obs,np.eye(n_agents)], axis=1)
    q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs))
    actions = np.argmax(q_values.detach().numpy(), axis=1)

    obs, reward, done, _ = env.step(actions)
    if all(done) == True:
         break
