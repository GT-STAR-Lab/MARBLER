import copy
import yaml
import numpy as np

def main():
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    agents={}
    agents['predator'] = {}
    agents['capture'] = {}
    agents['test_predator'] = {}
    agents['test_capture'] = {}
    num_candidates = config['n_capture_agents'] + config['n_test_capture_agents'] + config['n_predator_agents'] + config['n_test_predator_agents']
    idx_size = int(np.ceil(np.log2(num_candidates)))

    candidate = 0

    func_args = copy.deepcopy(config['traits']['capture'])
    del func_args['distribution']    
    for i in range(config['n_capture_agents']):
        agents['capture'][i] = {}
        agents['capture'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['capture']['distribution'])(**func_args)
        agents['capture'][i]['capture_radius'] = float(val)
        candidate += 1

    func_args = copy.deepcopy(config['traits']['predator'])
    del func_args['distribution']    
    for i in range(config['n_predator_agents']):
        agents['predator'][i] = {}
        agents['predator'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['predator']['distribution'])(**func_args)
        agents['predator'][i]['sensing_radius'] = float(val)
        candidate += 1

    func_args = copy.deepcopy(config['traits']['capture'])
    del func_args['distribution']
    for i in range(config['n_test_capture_agents']):
        agents['test_capture'][i] = {}
        agents['test_capture'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['capture']['distribution'])(**func_args)
        agents['test_capture'][i]['capture_radius'] = float(val)
        candidate += 1

    func_args = copy.deepcopy(config['traits']['predator'])
    del func_args['distribution']
    for i in range(config['n_test_predator_agents']):
        agents['test_predator'][i] = {}
        agents['test_predator'][i]['id'] = format(candidate, '#0'+str(idx_size + 2)+'b').replace('0b', '')
        val = getattr(np.random, config['traits']['predator']['distribution'])(**func_args)
        agents['test_predator'][i]['sensing_radius'] = float(val)
        candidate += 1

    with open('predefined_agents.yaml', 'w') as outfile:
        yaml.dump(agents, outfile, default_flow_style=False, allow_unicode=True)

if __name__ == '__main__':
    main()