import cv2
import copy
import json
import os
import torch
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

class Logger:
    def __init__(self, config, curriculum_stage=0):
        self.config = config
        
        self._resetValues()
        self._setupDirectories(curriculum_stage)
        self._saveConfig()

    def _resetValues(self):
        self.agent_actor_loss = [([], []) for _ in range(self.config.num_agents)]
        self.agent_critic_loss = [([], []) for _ in range(self.config.num_agents)]
        self.agent_global_reward_estimator_loss = [([], []) for _ in range(self.config.num_agents)]
        self.agent_training_reward = [([], []) for _ in range(self.config.num_agents)]
        self.agent_evaluation_reward = [([], []) for _ in range(self.config.num_agents)]
        self.agent_weight_stats = ([], [])

        if self.config.is_adversarial_environment:
            self.adversary_actor_loss = [([], []) for _ in range(self.config.num_adversaries)]
            self.adversary_critic_loss = [([], []) for _ in range(self.config.num_adversaries)]
            self.adversary_global_reward_estimator_loss = [([], []) for _ in range(self.config.num_adversaries)]
            self.adversary_training_reward = [([], []) for _ in range(self.config.num_adversaries)]
            self.adversary_evaluation_reward = [([], []) for _ in range(self.config.num_adversaries)]
            self.adversary_weight_stats = ([], [])

        if self.config.iterative_disruptor_training:
            self.disruptor_actor_loss = ([], [])
            self.disruptor_critic_loss = ([], [])

    def initVideos(self, episode):
        if not self.config.log:
            return
        self.training_video = cv2.VideoWriter(self.video_path + "/training_" + str(episode) + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (700, 700))
        self.evaluation_video = cv2.VideoWriter(self.video_path + "/evaluation_" + str(episode) + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (700, 700))

    def logTrainingFrame(self, rgb_array):
        if not self.config.log:
            return
        self.training_video.write(rgb_array)

    def logEvaluationFrame(self, rgb_array):
        if not self.config.log:
            return
        self.evaluation_video.write(rgb_array)

    def logTrainingRewards(self, rewards):
        if not self.config.is_rollout_algorithm:
            for id, reward in rewards.items():
                ix = int(id.split("_")[-1])
                if "agent" in id:
                    self.agent_training_reward[ix][1].append(reward)
                elif "adversary" in id:
                    self.adversary_training_reward[ix][1].append(reward)
            if len(self.agent_training_reward[0][1]) == self.config.steps:
                for ix in range(len(self.agent_training_reward)):
                    self.agent_training_reward[ix][0].append(sum(self.agent_training_reward[ix][1]))
                    self.agent_training_reward[ix][1].clear()
                if self.config.is_adversarial_environment:
                    for ix in range(len(self.adversary_training_reward)):
                        self.adversary_training_reward[ix][0].append(sum(self.adversary_training_reward[ix][1]))
                        self.adversary_training_reward[ix][1].clear()
        else:
            rewards = copy.deepcopy(rewards)
            rewards = {id: [dic[id] for dic in rewards] for id in rewards[0]}
            for id, reward in rewards.items():
                ix = int(id.split("_")[-1])
                if "agent" in id:
                    self.agent_training_reward[ix][1].extend(reward)
                elif "adversary" in id:
                    self.adversary_training_reward[ix][1].extend(reward)
            for ix in range(len(self.agent_training_reward)):
                for rollout in range(self.config.rollouts):
                    self.agent_training_reward[ix][0].append(sum(self.agent_training_reward[ix][1][rollout * self.config.steps:(rollout+1) * self.config.steps]))
                self.agent_training_reward[ix][1].clear()
            if self.config.is_adversarial_environment:
                for ix in range(len(self.adversary_training_reward)):
                    for rollout in range(self.config.rollouts):
                        self.adversary_training_reward[ix][0].append(sum(self.adversary_training_reward[ix][1][rollout * self.config.steps:(rollout+1) * self.config.steps]))
                    self.adversary_training_reward[ix][1].clear()

    def logEvaluationRewards(self, rewards):
        for id, reward in rewards.items():
            ix = int(id.split("_")[-1])
            if "agent" in id:
                self.agent_evaluation_reward[ix][1].append(reward)
            elif "adversary" in id:
                self.adversary_evaluation_reward[ix][1].append(reward)
        if len(self.agent_evaluation_reward[0][1]) == self.config.steps * self.config.evaluation_episodes:
            for ix in range(len(self.agent_evaluation_reward)):
                self.agent_evaluation_reward[ix][0].append(sum(self.agent_evaluation_reward[ix][1]) / (self.config.evaluation_episodes))
                self.agent_evaluation_reward[ix][1].clear()
            if self.config.is_adversarial_environment:
                for ix in range(len(self.adversary_evaluation_reward)):
                    self.adversary_evaluation_reward[ix][0].append(sum(self.adversary_evaluation_reward[ix][1]) / (self.config.evaluation_episodes))
                    self.adversary_evaluation_reward[ix][1].clear()

    def logLosses(self, losses, disruptor=False):
        if not self.config.log:
            return
        if not self.config.is_rollout_algorithm:
            for id, loss in losses.items():
                ix = int(id.split("_")[-1])
                if "agent" in id:
                    self.agent_actor_loss[ix][1].append(loss[0])
                    self.agent_critic_loss[ix][1].append(loss[1])
                    if self.config.algorithm == "variant_two":
                        self.agent_global_reward_estimator_loss[ix][1].append(loss[2])
                elif "adversary" in id:
                    self.adversary_actor_loss[ix][1].append(loss[0])
                    self.adversary_critic_loss[ix][1].append(loss[1])
                    if self.config.algorithm == "variant_two":
                        self.adversary_global_reward_estimator_loss[ix][1].append(loss[2])
            if len(self.agent_actor_loss[0][1]) == self.config.steps:
                for ix in range(len(self.agent_actor_loss)):
                    self.agent_actor_loss[ix][0].append(sum(self.agent_actor_loss[ix][1]) / self.config.steps)
                    self.agent_critic_loss[ix][0].append(sum(self.agent_critic_loss[ix][1]) / self.config.steps)
                    self.agent_actor_loss[ix][1].clear()
                    self.agent_critic_loss[ix][1].clear()
                    if self.config.algorithm == "variant_two":
                        self.agent_global_reward_estimator_loss[ix][0].append(sum(self.agent_global_reward_estimator_loss[ix][1]) / self.config.steps)
                        self.agent_global_reward_estimator_loss[ix][1].clear()
                if self.config.is_adversarial_environment:
                    for ix in range(len(self.adversary_actor_loss)):
                        self.adversary_actor_loss[ix][0].append(sum(self.adversary_actor_loss[ix][1]) / self.config.steps)
                        self.adversary_critic_loss[ix][0].append(sum(self.adversary_critic_loss[ix][1]) / self.config.steps)
                        self.adversary_actor_loss[ix][1].clear()
                        self.adversary_critic_loss[ix][1].clear()
                        if self.config.algorithm == "variant_two":
                            self.adversary_global_reward_estimator_loss[ix][0].append(sum(self.adversary_global_reward_estimator_loss[ix][1]) / self.config.steps)
                            self.adversary_global_reward_estimator_loss[ix][1].clear()
        else:
            if disruptor:
                self.disruptor_actor_loss[0].append(losses[0])
                self.disruptor_critic_loss[0].append(losses[1])
            else:
                for id, loss in losses.items():
                    ix = int(id.split("_")[-1])
                    if "agent" in id:
                        self.agent_actor_loss[ix][0].append(loss[0])
                        self.agent_critic_loss[ix][0].append(loss[1])
                    elif "adversary" in id:
                        self.adversary_actor_loss[ix][0].append(loss[0])
                        self.adversary_critic_loss[ix][0].append(loss[1])

    def logActorWeightStats(self, actor_weight_stats):
        if not self.config.log:
            return
        self.agent_weight_stats[0].append(actor_weight_stats)
        if self.config.is_adversarial_environment:
            self.adversary_weight_stats[0].append(actor_weight_stats)

    def printTrainingStep(self, step, rollout=None):
        if not self.config.is_rollout_algorithm:
            print(" |  Step: " + str(step + 1), end="\r")
        else:
            print(" |  Rollout: " + str(rollout + 1) +  "    Step: " + str(step + 1), end="\r")

    def printEvaluationStep(self, step, evaluation_episode):
        print(" |  Mini Episode: " + str(evaluation_episode + 1) + "    Step: " + str(step + 1), end="\r")

    def printTrainingEpisode(self, episode):
        last_agent_reward = sum([x[0][-1] for x in self.agent_training_reward]) / self.config.num_agents
        if self.config.is_adversarial_environment:
            last_adversary_reward = sum([x[0][-1] for x in self.adversary_training_reward]) / self.config.num_adversaries
            print(" |  Episode: " + str(episode) + "   Agent Reward: " + "%.8f" % round(last_agent_reward, 8) + "   Adversary Reward: " + "%.8f" % round(last_adversary_reward, 8))
        else:
            print(" |  Episode: " + str(episode) + "   Reward: " + "%.8f" % round(last_agent_reward, 8))

    def printEvaluationEpisode(self, episode):
        last_agent_reward = sum([x[0][-1] for x in self.agent_evaluation_reward]) / self.config.num_agents
        if self.config.is_adversarial_environment:
            last_adversary_reward = sum([x[0][-1] for x in self.adversary_evaluation_reward]) / self.config.num_adversaries
            print(" +  Episode: " + str(episode) + "   Agent Reward: " + "%.8f" % round(last_agent_reward, 8) + "   Adversary Reward: " + "%.8f" % round(last_adversary_reward, 8))
        else:
            print(" +  Episode: " + str(episode) + "   Reward: " + "%.8f" % round(last_agent_reward, 8))
    
    def savePlots(self, episode):
        if not self.config.log:
            return
        agent_average_training_reward = []
        agent_average_evaluation_reward = []
        for i in range(len(self.agent_training_reward[0][0])):
            agent_average_training_reward.append(sum([reward[0][i] for reward in self.agent_training_reward]) / self.config.num_agents)
        for i in range(len(self.agent_evaluation_reward[0][0])):
            agent_average_evaluation_reward.append(sum([reward[0][i] for reward in self.agent_evaluation_reward]) / self.config.num_agents)
        if self.config.is_adversarial_environment:
            adversary_average_training_reward = []
            adversary_average_evaluation_reward = []
            for i in range(len(self.adversary_training_reward[0][0])):
                adversary_average_training_reward.append(sum([reward[0][i] for reward in self.adversary_training_reward]) / self.config.num_adversaries)
            for i in range(len(self.adversary_evaluation_reward[0][0])):
                adversary_average_evaluation_reward.append(sum([reward[0][i] for reward in self.adversary_evaluation_reward]) / self.config.num_adversaries)
        
        if self.config.is_adversarial_environment:
            self._plotPerAgent(self.agent_actor_loss, episode, "Agent Actor Loss", "Episodes", "Loss", "agent_actor_loss")
            self._plotPerAgent(self.agent_critic_loss, episode, "Agent Critic Loss", "Episodes", "Loss", "agent_critic_loss")
            self._plotPerAgent(self.adversary_actor_loss, episode, "Adversary Actor Loss", "Episodes", "Loss", "adversary_actor_loss")
            self._plotPerAgent(self.adversary_critic_loss, episode, "Adversary Critic Loss", "Episodes", "Loss", "adversary_critic_loss")
            if self.config.algorithm == "variant_two":
                self._plotPerAgent(self.agent_global_reward_estimator_loss, episode, "Agent Global Reward Estimator Loss", "Episodes", "Loss", "agent_global_reward_estimator_loss")
                self._plotPerAgent(self.adversary_global_reward_estimator_loss, episode, "Adversary Global Reward Estimator Loss", "Episodes", "Loss", "adversary_global_reward_estimator_loss")
            self._plotAverage(agent_average_training_reward, episode, "Agent Training Reward", "Episodes", "Loss", "purple", "agent_training_reward")
            self._plotAverage(agent_average_evaluation_reward, episode, "Agent Evaluation Reward", "Episodes", "Loss", "green", "agent_evaluation_reward")
            self._plotAverage(adversary_average_training_reward, episode, "Adversary Training Reward", "Episodes", "Loss", "purple", "adversary_training_reward")
            self._plotAverage(adversary_average_evaluation_reward, episode, "Adversary Evaluation Reward", "Episodes", "Loss", "green", "adversary_evaluation_reward")
        else:
            self._plotPerAgent(self.agent_actor_loss, episode, "Actor Loss", "Episodes", "Loss", "actor_loss")
            self._plotPerAgent(self.agent_critic_loss, episode, "Critic Loss", "Episodes", "Loss", "critic_loss")
            self._plotWeightStats(self.agent_weight_stats, episode, "Actor First Layer Weights", "Episodes", "Sum of Weights", "actor_weight_stats")
            if self.config.iterative_disruptor_training:
                self._plotDisruptor(self.disruptor_actor_loss, episode, "Disruptor Actor Loss", "Episodes", "Loss", "orange", "disruptor_actor_loss")
                self._plotDisruptor(self.disruptor_critic_loss, episode, "Disruptor Critic Loss", "Episodes", "Loss", "orange", "disruptor_critic_loss")
            if self.config.algorithm == "variant_two":
                self._plotPerAgent(self.agent_global_reward_estimator_loss, episode, "Global Reward Estimator Loss", "Episodes", "Loss", "global_reward_estimator_loss")
            self._plotAverage(agent_average_training_reward, episode, "Training Reward", "Episodes", "Loss", "purple", "training_reward")
            self._plotAverage(agent_average_evaluation_reward, episode, "Evaluation Reward", "Episodes", "Loss", "green", "evaluation_reward")

    def saveVideos(self):
        if not self.config.log:
            return
        self.training_video.release()
        self.evaluation_video.release()

    def saveModels(self, models):
        if not self.config.log:
            return
        for id, models in models.items():
            torch.save(models[0].state_dict(), os.path.join(self.models_path, "actor_" + str(id) + ".pth"))
            torch.save(models[1].state_dict(), os.path.join(self.models_path, "critic_" + str(id) + ".pth"))
            if self.config.algorithm == "variant_two":
                torch.save(models[2].state_dict(), os.path.join(self.models_path, "global_reward_estimator_" + str(id) + ".pth"))

    def getRunPath(self):
        return self.run_path

    def _plotPerAgent(self, data, length, title, x_label, y_label, filename):
        figure, axis = plt.subplots(figsize=(12, 8))
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        for ix, agent_data in enumerate(data):
            if len(data):
                agent_data = agent_data[0]
                np.savetxt(os.path.join(self.values_path, filename + "_" + str(ix) + ".csv"), agent_data, delimiter=",")
                axis.plot(np.linspace(0, length, len(agent_data)), agent_data)
            else:
                axis.plot(0, 0)
        figure.savefig(os.path.join(self.plots_path, filename))
        plt.close(figure)

    def _plotWeightStats(self, data, length, title, x_label, y_label, filename):
        figure, axis = plt.subplots(figsize=(12, 8))
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        data = data[0]
        if len(data):
            individual_stats = [x[0] for x in data]
            landmark_stats = [x[1] for x in data]
            communication_stats = [x[2] for x in data]
            np.savetxt(os.path.join(self.values_path, filename + ".csv"), data, delimiter=",")
            axis.plot(np.linspace(0, length, len(data)), individual_stats, label="Individual Information Weights")
            axis.plot(np.linspace(0, length, len(data)), communication_stats, label="Landmark Information Weights")
            axis.plot(np.linspace(0, length, len(data)), landmark_stats, label="Communication Information Weights")
            axis.legend()
        else:
            axis.plot(0, 0)
        figure.savefig(os.path.join(self.plots_path, filename))
        plt.close(figure)

    def _plotDisruptor(self, data, length, title, x_label, y_label, color, filename):
        figure, axis = plt.subplots(figsize=(12, 8))
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        if len(data):
            np.savetxt(os.path.join(self.values_path, filename + ".csv"), data[0], delimiter=",")
            axis.plot(np.linspace(0, length, len(data[0])), data[0], color=color)
        else:
            axis.plot(0, 0)
        figure.savefig(os.path.join(self.plots_path, filename))
        plt.close(figure)

    def _plotAverage(self, data, length, title, x_label, y_label, color, filename):
        figure, axis = plt.subplots(figsize=(12, 8))
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        if len(data):
            np.savetxt(os.path.join(self.values_path, filename + ".csv"), data, delimiter=",")
            axis.plot(np.linspace(0, length, len(data)), data, color=color)
        else:
            axis.plot(0, 0)
        figure.savefig(os.path.join(self.plots_path, filename))
        plt.close(figure)
    
    def _setupDirectories(self, curriculum_stage=0):
        if not self.config.log:
            return

        if self.config.run_name:
            run_name = self.config.run_name
        else:
            run_name = datetime.now().strftime("%d-%m-%Y")

        if self.config.curriculum_learning:
            run_name = f"curriculum_stage_{curriculum_stage}_" + run_name

        self.run_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "runs", run_name))
            
        version = 1
        while os.path.exists(self.run_path):
            self.run_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "runs", run_name + "_" + str(version)))
            version += 1

        self.video_path = os.path.join(self.run_path, "video")
        self.plots_path = os.path.join(self.run_path, "plots")
        self.values_path = os.path.join(self.run_path, "plots/values")
        self.models_path = os.path.join(self.run_path, "models")
        
        os.makedirs(self.run_path)
        os.makedirs(self.video_path)
        os.makedirs(self.plots_path)
        os.makedirs(self.values_path)
        os.makedirs(self.models_path)

    def _saveConfig(self):
        if not self.config.log:
            return
        json_dict = {}
        for attribute in vars(self.config):
            json_dict[attribute] = getattr(self.config, attribute)
        json_object = json.dumps(json_dict, indent=4)
        with open(os.path.join(self.run_path, "config.json"), "w") as file:
            file.write(json_object)

