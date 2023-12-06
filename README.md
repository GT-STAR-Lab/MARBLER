# MARBLER: Multi-Agent RL Benchmark and Learning Environment for the Robotarium
Team: Reza Torbati, Shubham Lohiya, Shivika Singh, Meher Nigam

## Installation Instructions
1. Create new Conda Environment: `conda create -n MARBLER python=3.8 && conda activate MARBLER`. 
- Note that python 3.8 is only chosen to ensure compatitbility with EPyMARL.
2. Download and Install the [Robotarium Python Simulator](https://github.com/robotarium/robotarium_python_simulator)
- As of now, the most recent commit our code works with is 6bb184e. The code will run with the most recent push to the Robotarium but it will crash during training.
3. Install our environment by running `pip install -e .` in this directory
4. To test successfull installation, run `python3 -m robotarium_gym.main` to run a pretrained model

## Usage
* To look at current scenarios or create new ones or to evaluate trained models, look at the README in robotarium_gym
* To upload the agents to the Robotarium, look at the README in robotarium_eval

## Training with EPyMARL
1. Download and Install [EPyMARL](https://github.com/uoe-agents/epymarl). On Ubuntu 22.04, to successfully install EPyMARL, I have to: 
    - Change line 15 in `requirements.txt` from `protobuf==3.6.1` to `protobuf`
    - Downgrade `wheel` to 0.38.4
    - Downgrade `setuptools` to 65.5.0
    - Install `einops` and `torchscatter`
2. Train agents normally using our gym keys
- For example: `python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePrey-v0"`
- To train faster, ensure `robotarium` is False, `real_time` is False, and `show_figure_frequency` is large or -1 in the environment's `config.yaml`
- Known error: if `env_args.time_limit<max_episode_steps`, EPyMARL will crash after the first episode
3. Copy the trained weights to the models folder for the scenario that was trained
- Requires the agent.th file (location should be printed in the cout of the terminal the model was trained in, typically in EPyMARL/results/models/...)
- Requires the config.json file (typically in EPyMARL/results/algorithm_name/gym:scenario/...)
4. Update the scenario's config.yaml to use the newly trained agents


## Citing
If you use this in your work please cite:
* Our work:
>Reza Torbati, Shubham Lohiya, Shivika Singh, Meher Shashwat Nigam, & Harish Ravichandar. (2023). MARBLER: An Open Platform for Standarized Evaluation of Multi-Robot Reinforcement Learning Algorithms.
* The Robotarium: 
>S. Wilson, P. Glotfelter, L. Wang, S. Mayya, G. Notomista, M. Mote, and M. Egerstedt. The robotarium: Globally impactful opportunities, challenges, and lessons learned in remote-access, distributed control of multirobot systems. IEEE Control Systems Magazine, 40(1):26–44, 2020.
* Additionally, if you use the default agents in this repo, also cite EPyMARL:
>Papoudakis, Georgios, et al. "Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks." arXiv preprint arXiv:2006.07869 (2020).
