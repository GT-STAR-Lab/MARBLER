import copy
import yaml
import numpy as np

def main():
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    agents={}
    agents['train'] = {}
    agents['test'] = {}
    num_candidates = config['n_train_agents'] + config['n_test_agents']
    idx_size = int(np.ceil(np.log2(num_candidates)))

    func_args = copy.deepcopy(config['traits']['torque'])
    del func_args['distribution']   

    candidate = 0
    for i in range(config['n_train_agents']):
        agents['train'][i] = {}
        agents['train'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = round(getattr(np.random, config['traits']['torque']['distribution'])(**func_args))
        agents['train'][i]['torque'] = float(val)
        candidate += 1
 
    for i in range(config['n_test_agents']):
        agents['test'][i] = {}
        agents['test'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = round(getattr(np.random, config['traits']['torque']['distribution'])(**func_args))
        agents['test'][i]['torque'] = float(val)
        candidate += 1

    with open('predefined_agents.yaml', 'w') as outfile:
        yaml.dump(agents, outfile, default_flow_style=False, allow_unicode=True)

if __name__ == '__main__':
    main()