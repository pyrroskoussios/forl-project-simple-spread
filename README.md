# **Not so simple: an assessment of the simple spread multi particle environment and communicative MARL**

## Introduction

This repository, contains all code used to generate results for our FoRL project on communicative MARL on the simple spread benchmark. We include a fork of the PettingZoo repository (__https://github.com/Farama-Foundation/PettingZoo__), as multiple adjustments to the environment were made.

## Project Structure

All code written by us can be found under the `src/` directory. It is structured as follows:

```
src/
├── default_config.json
├── agents/
│   ├── algorithms/
│   │   ├── ppo.py
│   │   ├── variant_one.py
│   │   └── variant_two.py
│   ├── networks/
│   │   ├── mlp.py
│   │   └── rnn.py
│   ├── disruptor.py
│   └── manager.py
├── arguments.py
├── environment.py
├── logger.py
├── main.py
└── default_config.json
```


A brief breakdown of the files and directories responsibilities:

- `main.py`
    The main entry point of the program and executes/coordinates the training, evaluating and logging.


- `default_config.json`
    The default parameters that are used when main.py is run without specifying any arguments via CLI.


- `logger.py`
    Responsible for logging training plots, videos and models. It saves every plot with the values, allowing for replotting once the training run is finished. Additionally, it saves a `config.json` file containing the parameters used for that run, allowing for training to be continued at a later point in time.
    
- `arguments.py`
    Parses CLI arguments and can load saved `config.json` files. Please note that numerous parameters can be specified, resulting in an extremely large number of possible permutations. We have tried to ensure the most egregious false-input cases are checked, however it is likely that certain combinations may lead to undefined behavior.

- `environment.py`
    Serves as a wrapper for the forked PettingZoo environment.

- `agents/algorithms/`
    Contains the trainable algorithms. This includes the main PPO implementation aswell as \"variant_one\" and \"variant_two\" from \"Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents\" by Zhang et al. (__https://arxiv.org/abs/1802.08757__).

- `agents/newtorks/`
    Contains simple MLP and GRU Network implementations for use as actor and critic networks.

- `agents/disruptor.py`
    Contains the disruptor PPO implementation, which is practically identical to the regular one but is designed to minimize the summed agent reward.

- `agents/manager.py`
    Deals with the multiple agents, allowing `main.py` to call step once without having to call it for every agent.

## Quick Start Guide

To begin a decentralized MARL training run of your own, start by cloning the repository:
```console
$ git clone https://github.com/pyrroskoussios/forl-project-simple-spread.git
```
After entering your python development environment, install the requirements:
```console
$ pip3 install -r requirements.txt
```
Now install the PettingZoo fork:
```console
$ pip3 install forl-project-simple-spread/libs/PettingZoo
```
Now you can run `main.py` using the arguments from `default_config.json` and with logging enabled, which will create the directory `forl-project-simple-spread/runs` and save your training run inside it:
```console
$ python3 forl-project-simple-spread/src/main.py --log True
```


Please reach out if anything does not work :D


