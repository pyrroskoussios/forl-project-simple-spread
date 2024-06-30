import torch
import random
import numpy as np

from agents.manager import AgentManager
from environment import Environment
from arguments import Config
from logger import Logger

class Trainer:
    def __init__(self, config, curriculum_stage=0):
        self.config = config
        self.seed()
        self.logger = Logger(self.config, curriculum_stage)
        self.environment = Environment(self.config)
        action_space = self.environment.getActionSpace()
        observation_space = self.environment.getObservationSpace()

        agent_ids = self.environment.getAgentIds()
        self.agents = AgentManager(self.config, observation_space, action_space, agent_ids)

    def seed(self):
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def trainingEpisode(self, train_disruptor, log_video=False, first_sequence=False):
        if not self.config.is_rollout_algorithm:
            self.agents.resetEpisodicValues()
            observations, _ = self.environment.reset()
            done = False
            step = 0
            while not done:
                actions, _ = self.agents.chooseActions(observations)
                next_observations, rewards, _, truncations, _ = self.environment.step(actions)
                losses = self.agents.updateAC(observations, actions, next_observations, rewards)
                self.logger.printTrainingStep(step)
                self.logger.logTrainingRewards(rewards)
                self.logger.logLosses(losses)
                if log_video:
                    self.logger.logTrainingFrame(self.environment.render())
                done = all(truncated for truncated in truncations.values())
                observations = next_observations
                step += 1
        else:
            rollout_data = {
                "states": [],
                "actions": [],
                "probabilities": [],
                "rewards": [],
            }
            for rollout in range(self.config.rollouts):
                observations, _ = self.environment.reset()
                done = False
                step = 0
                while not done:
                    if train_disruptor and not first_sequence: 
                        disruptor_action, disruptor_probabilities, disruptor_observation, observations = self.agents.disruptObservations(observations, exploit=not train_disruptor)
                        actions, probabilities = self.agents.chooseActions(observations, exploit=train_disruptor)
                    else:
                        actions, probabilities = self.agents.chooseActions(observations)

                    next_observations, rewards, _, truncations, _ = self.environment.step(actions)
                    if train_disruptor:
                        rollout_data["states"].append(disruptor_observation)
                        rollout_data["actions"].append(disruptor_action)
                        rollout_data["probabilities"].append(disruptor_probabilities)
                    else:
                        rollout_data["states"].append(observations)
                        rollout_data["actions"].append(actions)
                        rollout_data["probabilities"].append(probabilities)
                    rollout_data["rewards"].append(rewards)
                    self.logger.printTrainingStep(step, rollout)
                    if log_video:
                        self.logger.logTrainingFrame(self.environment.render())
                    done = all(truncated for truncated in truncations.values())
                    observations = next_observations
                    step += 1
            if train_disruptor:
                disruptor_losses = self.agents.updateDisruptor(rollout_data)
                self.logger.logLosses(disruptor_losses, disruptor=True)
            else:
                agent_losses = self.agents.updatePPO(rollout_data)
                self.logger.logLosses(agent_losses)
                actor_weight_stats = self.agents.getActorWeightStats()
                self.logger.logActorWeightStats(actor_weight_stats)
            self.logger.logTrainingRewards(rollout_data["rewards"])

    def evaluationEpisode(self, use_disruptor, evaluation_episode, log_video):
        if not self.config.is_rollout_algorithm:
            self.agents.resetEpisodicValues()
            observations, _ = self.environment.reset()
            done = False
            step = 0
            while not done:
                actions, _ = self.agents.chooseActions(observations, exploit=True)
                next_observations, rewards, _, truncations, _ = self.environment.step(actions)
                self.logger.printEvaluationStep(step, evaluation_episode)
                self.logger.logEvaluationRewards(rewards)
                if log_video:
                    self.logger.logEvaluationFrame(self.environment.render())
                done = all(truncated for truncated in truncations.values())
                observations = next_observations
                step += 1
        else:
            observations, _ = self.environment.reset()
            done = False
            step = 0
            while not done:
                if use_disruptor:
                    _, _, _, observations = self.agents.disruptObservations(observations, exploit=True)
                actions, _ = self.agents.chooseActions(observations, exploit=True)
                next_observations, rewards, _, truncations, _ = self.environment.step(actions)
                self.logger.printEvaluationStep(step, evaluation_episode)
                self.logger.logEvaluationRewards(rewards)
                if log_video:
                    self.logger.logEvaluationFrame(self.environment.render())
                done = all(truncated for truncated in truncations.values())
                observations = next_observations
                step += 1

    def trainingLoop(self):
        evaluation_frequency = self.config.evaluation_frequency * (self.config.rollouts if self.config.is_rollout_algorithm else 1)
        for episode in range(0, self.config.episodes, self.config.rollouts if self.config.is_rollout_algorithm else 1):
            iteration = episode // self.config.rollouts if self.config.is_rollout_algorithm else 1
            train_disruptor = bool(iteration // self.config.iteration_length % 2) if self.config.iterative_disruptor_training else False
            end_of_disruptor_iteration = bool(iteration and not (iteration % self.config.iteration_length) and not ((iteration // self.config.iteration_length) % 2)) if self.config.iterative_disruptor_training else False
            log_video = False
            if not episode % evaluation_frequency or episode == self.config.episodes - 1:
                log_video = True
                self.logger.initVideos(episode)
                self.logger.savePlots(episode)
                self.logger.saveModels(self.agents.getModels())

            self.trainingEpisode(train_disruptor, log_video, first_sequence=iteration < self.config.iteration_length)
            self.logger.printTrainingEpisode(episode)
            if not episode % evaluation_frequency:
                for evaluation_episode in range(self.config.evaluation_episodes):
                    self.evaluationEpisode(train_disruptor, evaluation_episode, log_video)
                self.logger.printEvaluationEpisode(episode)
            if log_video:
                self.logger.saveVideos()
            if end_of_disruptor_iteration:
                self.agents.addToDisruptorSet()

    def __call__(self):
        if self.config.curriculum_learning:
            for stage in range(self.config.curriculum_stages + 1):
                if stage > 0:
                    last_stage_path = self.logger.getRunPath()
                    self.config.num_agents += 1
                    self.__init__(self.config, stage)
                    self.agents.loadAgents(last_stage_path, curriculum_load=True)
                self.trainingLoop()
        else:
            self.trainingLoop()


if  __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer()
