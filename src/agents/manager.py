import os
import copy
import torch
import random
import numpy as np

from agents.algorithms.variant_one import VariantOne
from agents.algorithms.variant_two import VariantTwo
from agents.algorithms.ppo import PPO
from agents.disruptor import Disruptor

class AgentManager:
    def __init__(self, config, observation_space, action_space, agent_ids):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        
        algorithm_class = {
            "variant_one": VariantOne,
            "variant_two": VariantTwo,
            "ppo": PPO
        }
        
        agents = [algorithm_class[self.config.algorithm](config, observation_space["agents"], action_space["agents"]) for _ in range(self.config.num_agents)]
        if self.config.is_adversarial_environment:
            adversaries = [algorithm_class[self.config.algorithm](config, observation_space["adversaries"], action_space["adversaries"]) for _ in range(self.config.num_adversaries)]
        else:
            adversaries = []

        if self.config.iterative_disruptor_training:
            self.disruptor = Disruptor(config, observation_space["agents"], action_space["agents"])
            self.disruptor_policies = []
        
        if self.config.centralized:
            central_agent_critic = agents[0].critic
            for agent in agents:
                agent.critic = central_agent_critic
            if self.config.is_adversarial_environment:
                central_adversary_critic = adversaries[0].critic
                for adversary in adversaries:
                    adversary.critic = central_adversary_critic

            central_agent_actor = agents[0].actor
            for agent in agents:
                agent.actor = central_agent_actor
        
        self.agents = dict(zip(agent_ids, adversaries + agents)) 

    def resetEpisodicValues(self):
        for _, agent in self.agents.items():
            agent.resetEpisodicValues()

    def getModels(self):
        models = {}
        for id, agent in self.agents.items():
            if self.config.algorithm == "variant_two":
                models[id] = (agent.actor, agent.critic, agent.global_reward_estimator)
            else:
                models[id] = (agent.actor, agent.critic)
        return models

    def getActorWeightStats(self):
        actor_weight_stats = [0, 0, 0]
        for _, agent in self.agents.items():
            stats = agent.getActorWeightStats()
            for ix in range(3):
                actor_weight_stats[ix] += stats[ix] / self.config.num_agents
        actor_weight_stats = tuple(actor_weight_stats)
        return actor_weight_stats

    def addToDisruptorSet(self):
        self.disruptor_policies.append(self.disruptor)
        self.disruptor = Disruptor(self.config, self.observation_space["agents"], self.action_space["agents"])

    def chooseActions(self, observations, exploit=False):
        actions = {}
        probabilites = {}
        for id, agent in self.agents.items():
            action, probability = agent.chooseAction(observations[id], exploit)
            actions[id] = action
            probabilites[id] = probability
        return actions, probabilites
    
    def disruptObservations(self, observations, exploit=False):
        agents_to_disrupt = np.random.choice(self.config.num_agents, self.config.disrupt_count, replace=False)
        for selected_agent in range(self.config.num_agents):
            if selected_agent not in agents_to_disrupt:
                continue
            selected_agent_tensor = torch.full((self.config.frame_stack, 1), selected_agent * (1 / (self.config.num_agents - 1)))
            disruptor_observation = torch.cat((selected_agent_tensor, observations["agent_" + str(selected_agent)]), 1)
            
            if exploit and len(self.disruptor_policies):
                disruptor_policy = random.choice(self.disruptor_policies) 
            else:
                disruptor_policy = self.disruptor
            
            action, log_probabilities, disruption = disruptor_policy.chooseDisruption(disruptor_observation, exploit=exploit)
            disruption_view = disruption.view(self.config.frame_stack, 2 * (self.config.num_agents - 1))
            new_observations = torch.cat((observations["agent_" + str(selected_agent)][:, :-2 * (self.config.num_agents - 1)], disruption_view), 1)
            observations["agent_" + str(selected_agent)] = new_observations
        return action, log_probabilities, disruptor_observation, observations
    
    def updateAC(self, observations, actions, next_observations, rewards):
        losses = {}
        for id, agent in self.agents.items():
            loss = agent.update(observations[id], actions[id], next_observations[id], rewards[id])
            losses[id] = loss

        if not self.config.centralized:
            self._communicate()

        return losses

    def updatePPO(self, rollout_data):
        rollout_data = copy.deepcopy(rollout_data)
        for name, data in rollout_data.items():
            rollout_data[name] = {id: [dic[id] for dic in data] for id in data[0]}
        
        losses = {}
        for id, agent in self.agents.items():
            agent_rollout_data = {name: rollout_data[name][id] for name in rollout_data}
            for name in ["states", "probabilities"]:
                agent_rollout_data[name] = torch.stack(agent_rollout_data[name])
            loss = agent.update(agent_rollout_data)
            losses[id] = loss
         
        return losses
    
    def updateDisruptor(self, rollout_data):
        rollout_data = copy.deepcopy(rollout_data)
        for ix, rewards in enumerate(rollout_data["rewards"]):
            rollout_data["rewards"][ix] = -sum(rewards.values()) / self.config.num_agents 
        for name in ["states", "actions", "probabilities"]:
            rollout_data[name] = torch.stack(rollout_data[name])
        losses = self.disruptor.update(rollout_data)
        return losses

    def loadAgents(self, path, curriculum_load=False):
        for ix, agent in self.agents.items():
            if curriculum_load and ix == "agent_" + str(len(self.agents) - 1):
                break
            agent.actor.load_state_dict(torch.load(os.path.join(path, "models", "actor_" + ix + ".pth")))
            agent.critic.load_state_dict(torch.load(os.path.join(path, "models", "critic_" + ix + ".pth")))

    def _communicate(self):
        weights = np.random.rand(self.config.num_agents, self.config.connectivity)
        normalized_weights = weights / weights.sum(axis=1)[:, np.newaxis]
        connectivity_matrix = np.pad(normalized_weights, ((0, 0), (self.config.num_agents - self.config.connectivity, 0)), "constant", constant_values=0)
        for row in connectivity_matrix:
            np.random.shuffle(row)

        for ix, (_, agent) in enumerate(self.agents.items()):
            agent_critic_weights = agent.getCriticWeights()
            for parameter in agent_critic_weights:
                agent_critic_weights[parameter] = torch.zeros_like(agent_critic_weights[parameter])
                for other_ix, (_, other_agent) in enumerate(self.agents.items()):
                    other_agent_critic_weights = other_agent.getCriticWeights()
                    agent_critic_weights[parameter] += connectivity_matrix[ix, other_ix] * other_agent_critic_weights[parameter]
            agent.setCriticWeights(agent_critic_weights)        
        
        if self.config.algorithm == "variant_two":
            for ix, (_, agent) in enumerate(self.agents.items()):
                agent_global_reward_estimator_weights = agent.getGlobalRewardEstimatorWeights()
                for parameter in agent_global_reward_estimator_weights:
                    agent_global_reward_estimator_weights[parameter] = torch.zeros_like(agent_global_reward_estimator_weights[parameter])
                    for other_ix, (_, other_agent) in enumerate(self.agents.items()):
                        other_agent_global_reward_estimator_weights = other_agent.getGlobalRewardEstimatorWeights()
                        agent_global_reward_estimator_weights[parameter] += connectivity_matrix[ix, other_ix] * other_agent_global_reward_estimator_weights[parameter]
                agent.setGlobalRewardEstimatorWeights(agent_global_reward_estimator_weights)        

            for ix, (_, agent) in enumerate(self.agents.items()):
                agent_long_term_reward = 0
                for other_ix, (_, other_agent) in enumerate(self.agents.items()):
                    other_agent_long_term_reward = other_agent.getLongTermReward()
                    agent_long_term_reward += connectivity_matrix[ix, other_ix] * other_agent_long_term_reward
                agent.setLongTermReward(agent_long_term_reward)        





    

