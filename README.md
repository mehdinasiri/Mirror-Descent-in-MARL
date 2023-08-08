# Heterogeneous Agent Mirror Descent Policy Optimization (HAMDPO)
Welcome to the GitHub repository for **HAMDPO**, a groundbreaking algorithm that extends mirror descent to Multi-Agent Reinforcement Learning. Our approach offers superior convergence and performance compared to state-of-the-art algorithms like [HATRPO](https://arxiv.org/pdf/2109.11251.pdf) and [HAPPO](https://arxiv.org/pdf/2109.11251.pdf) on StarCraft II and Multi-agent MUJOCO benchmarks. Harness the power of mirror descent to efficiently optimize agent policies and witness outstanding results in challenging environments. Feel free to explore the code, contribute, and be a part of shaping the future of Multi-Agent RL with HAMDPO. Your insights and contributions are highly valued!

## Installation
### Create environment
``` Bash
conda create -n env_name python=3.9
conda activate env_name
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

### Multi-agent MuJoCo
Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:
``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```
### StarCraft II & SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.

## How to run
When your environment is ready, you could run shell scripts provided. For example:
``` Bash
cd scripts
./train_mujoco.sh  # run with HAPPO/HATRPO on Multi-agent MuJoCo
./train_smac.sh  # run with HAPPO/HATRPO on StarCraft II
```

If you would like to change the configs of experiments, you could modify sh files or look for config files for more details. 
