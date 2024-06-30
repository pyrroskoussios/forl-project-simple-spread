import torch
import numpy as np

from copy import deepcopy
from collections import deque
from pettingzoo.mpe import simple_v3, simple_spread_v3, simple_reference_v3, simple_tag_v3

class Environment:
    def __init__(self, config):
        self.config = config

        if self.config.environment == "simple":
            self.env = simple_v3.parallel_env(
                render_mode="rgb_array",
                continuous_actions=False,
                max_cycles=self.config.steps,
            )
        elif self.config.environment == "simple_spread":
            self.env = simple_spread_v3.parallel_env(
                render_mode="rgb_array",
                continuous_actions=False,
                N=self.config.num_agents,
                max_cycles=self.config.steps,
                local_ratio=1
            )
        elif self.config.environment == "simple_reference":
            self.env = simple_reference_v3.parallel_env(
                render_mode="rgb_array",
                continuous_actions=False,
                max_cycles=self.config.steps,
                local_ratio=0.5
            )
        elif self.config.environment == "simple_tag":
            self.env = simple_tag_v3.parallel_env(
                render_mode="rgb_array",
                continuous_actions=False,
                num_good=self.config.num_agents,
                num_adversaries=self.config.num_adversaries,
                num_obstacles=0,
                max_cycles=self.config.steps
            )

        self.env.aec_env.env.env._seed(self.config.seed)

        self.env.aec_env.env.env.scenario.communication_reach = self.config.communication_reach
        self.env.aec_env.env.env.scenario.curriculum_learning = self.config.curriculum_learning
        self.env.aec_env.env.env.scenario.centralized = self.config.centralized

    def __del__(self):
        self.env.close()

    def getObservationSpace(self):
        if self.config.curriculum_learning and self.config.environment == "simple_spread":
            observation_space = {"agents": 14}
            if self.config.is_adversarial_environment:
                observation_space = {"adversaries": 14}
        else:
            observation_space = {"agents": self.env.aec_env.env.env.observation_spaces["agent_0"]._shape[0]}
            if self.config.is_adversarial_environment:
                observation_space["adversaries"] = self.env.aec_env.env.env.observation_spaces["adversary_0"]._shape[0]
        return observation_space

    def getActionSpace(self):
        action_space = {"agents": self.env.aec_env.env.env.action_spaces["agent_0"].n}
        if self.config.is_adversarial_environment:
            action_space["adversaries"] = self.env.aec_env.env.env.action_spaces["adversary_0"].n
        return action_space

    def getAgentIds(self):
        return self.env.possible_agents

    def reset(self):
        observations, infos = self.env.reset(np.random.randint(0, 2**31))
        if self.config.network_type == "rnn":
            self.observation_deque = deque(self.config.frame_stack * [observations], maxlen=self.config.frame_stack)
            observations = self._stack_observation_deque()
        observations = self._dictNumpyToTorch(observations) 
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        if self.config.centralized:
            summed_reward = sum(rewards.values())
            for key in rewards.keys():
                rewards[key] = summed_reward
        if self.config.network_type == "rnn":
            self.observation_deque.append(observations)
            observations = self._stack_observation_deque()
        observations = self._dictNumpyToTorch(observations)
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def _stack_observation_deque(self):
        observations = deepcopy(self.observation_deque[0])
        for id, _ in observations.items():
            observations[id] = np.array([x[id] for x in self.observation_deque])
        return observations

    def _dictNumpyToTorch(self, dict_numpy):
        dict_torch = dict((id, torch.FloatTensor(value).to(self.config.device)) for id, value in dict_numpy.items())
        return dict_torch

    def _dictTorchToNumpy(self, dict_torch):
        dict_numpy = dict((id, value.cpu().numpy()) for id, value in dict_torch.items())
        return dict_numpy




