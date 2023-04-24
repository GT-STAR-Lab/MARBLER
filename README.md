# MARBLER: Multi-Agent RL Benchmark and Learning Environment for the Robotarium
Team: Reza Torbati, Shubham Lohiya, Shivika Singh, Meher Nigam

## Installation Instructions
1. Create new Conda Environment: `conda create -n MARBLER python=3.8 && conda activate MARBLER`. 
- Note that python 3.8 is only chosen to ensure compatitbility with EPyMARL.
2. Download and Install the [Robotarium Python Simulator](https://github.com/robotarium/robotarium_python_simulator)
3. Install our environment by running `pip insatll -e .` in this directory
4. To test successfull installation run `python3 -m robotarium_gym.main` to run a pretrained model

## Training with EPyMARL
1. Download and Install [EPyMARL](https://github.com/uoe-agents/epymarl)
- In EPyMARL's `requirements.txt`, line 15 may need to be changed from `protobuf==3.6.1` to `protobuf`
2. Train agents normally using our gym keys
- For example: `python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePrey-v0")`
- To train faster, ensure `robotarium` is False, `real_time` is False, and `show_figure_frequency` is large or -1 in the environment's `config.yaml`
- Known error: if `env_args.time_limit<max_episode_steps`, EPyMARL will crash after the first episode

## Citations
* Entire backend of the code comes from: 
>S. Wilson, P. Glotfelter, L. Wang, S. Mayya, G. Notomista, M. Mote, and M. Egerstedt. The robotarium: Globally impactful opportunities, challenges, and lessons learned in remote-access, distributed control of multirobot systems. IEEE Control Systems Magazine, 40(1):26–44, 2020.

* Code Architecture Inspired By:
> I. Mordatch and P. Abbeel. Emergence of grounded compositional language in multi-agent populations. CoRR, abs/1703.04908, 2017. URL http://arxiv.org/abs/1703.04908.

* Agent Classes and Trained Weights Comes From:
> G. Papoudakis, F. Christianos, L. Sch ̈afer, and S. V. Albrecht. Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021. URL http://arxiv.org/abs/2006.07869.
