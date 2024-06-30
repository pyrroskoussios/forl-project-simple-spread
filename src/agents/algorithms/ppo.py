import torch
import numpy as np

from agents.networks.mlp import MLP
from agents.networks.rnn import RNN

class PPO:
    def __init__(self, config, observation_space, action_space):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        
        if self.config.network_type == "mlp":
            self.actor = MLP(observation_space, self.config.num_nodes, self.config.num_hidden_layers, action_space, output_activation="softmax").to(self.config.device)
            self.critic = MLP(observation_space, self.config.num_nodes, self.config.num_hidden_layers, 1).to(self.config.device)
        elif self.config.network_type == "rnn":
            self.actor = RNN(observation_space, self.config.num_nodes, self.config.num_hidden_layers, action_space, output_activation="softmax").to(self.config.device)
            self.critic = RNN(observation_space, self.config.num_nodes, self.config.num_hidden_layers, 1).to(self.config.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), eps=1e-5, lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), eps=1e-5, lr=self.config.critic_learning_rate)
        
    def chooseAction(self, state, exploit=False):
        with torch.no_grad():
            probability = self.actor(state)
        if exploit:
            action = probability.argmax().item()
        else:
            action = torch.multinomial(probability, 1).item()
        return action, probability 

    def getActorWeightStats(self):
        input_weights = self.actor.rnn_layers.weight_ih_l0.data
        communication_avg = input_weights[:, -(self.config.num_agents - 1) * 2:].abs().sum() / ((self.config.num_agents - 1) * 2)
        landmark_avg = input_weights[:, 4:4 + 2 * self.config.num_agents].abs().sum() / (self.config.num_agents * 2)
        individual_avg = input_weights[:, 0:4].abs().sum() / 4
        return individual_avg, landmark_avg, communication_avg

    def getCriticWeights(self):
        return self.critic.state_dict()

    def setCriticWeights(self, critic_weights):
        self.critic.load_state_dict(critic_weights)

    def update(self, rollout_data):
        average_actor_loss = 0
        average_critic_loss = 0
        batch_size = (self.config.steps * self.config.rollouts) // self.config.batches + ((self.config.steps * self.config.rollouts) % self.config.batches > 0)
        rollout_data["rewards"] = self._normalize_rewards(rollout_data["rewards"])
        for _ in range(self.config.epochs):
            for batch in range(self.config.batches):
                rollout_data_batch = {name: rollout_data[name][batch * batch_size:(batch + 1) * batch_size] for name in rollout_data}
                
                values = self.critic(rollout_data_batch["states"]).squeeze()
                advantages = self._calculateAdvantages(rollout_data_batch["rewards"], values)
                rewards_to_go_batch = advantages + values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                probabilities = self.actor(rollout_data_batch["states"])
                current_probability = []
                rollout_probability = []
                for ix in range(len(probabilities)):
                    current_probability.append(probabilities[ix][rollout_data_batch["actions"][ix]])
                    rollout_probability.append(rollout_data_batch["probabilities"][ix][rollout_data_batch["actions"][ix]])
                current_log_probability = torch.stack(current_probability).log()
                rollout_log_probability = torch.stack(rollout_probability).log()
                log_probability_ratio = torch.exp(current_log_probability - rollout_log_probability)
                surrogate_loss_one = log_probability_ratio * advantages
                surrogate_loss_two = torch.clamp(log_probability_ratio, 1 - self.config.clip_value, 1 + self.config.clip_value) * advantages
                actor_loss = (-torch.min(surrogate_loss_one, surrogate_loss_two)).mean()

                mse_loss = torch.nn.MSELoss()
                critic_loss = mse_loss(values, rewards_to_go_batch)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss = actor_loss.detach().cpu().item()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                critic_loss = critic_loss.detach().cpu().item()
                
                average_actor_loss += actor_loss / (self.config.batches * self.config.epochs)
                average_critic_loss += critic_loss / (self.config.batches * self.config.epochs)

        return (average_actor_loss, average_critic_loss)

    def _normalize_rewards(self, rewards):
        rewards = np.array(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        rewards = rewards.tolist()
        return rewards
    
    def _calculateAdvantages(self, rewards, values):
        gae_lambda = 0.99

        advantages = []
        for rollout in range(self.config.rollouts // self.config.batches):
            rollout_advantages = []
            last_advantage = 0
            rollout_rewards = rewards[rollout * self.config.steps:(rollout + 1) * self.config.steps]
            rollout_values = values[rollout * self.config.steps:(rollout + 1) * self.config.steps]
            for i in reversed(range(len(rollout_rewards))):
                if i == len(rollout_rewards) - 1:
                    delta = rollout_rewards[i] - rollout_values[i]
                    last_advantage = delta
                else:
                    delta = rollout_rewards[i] + self.config.discount_factor * rollout_values[i+1] - rollout_values[i]
                    last_advantage = delta + self.config.discount_factor * gae_lambda * last_advantage
                rollout_advantages.append(last_advantage) 
            advantages.append(list(reversed(rollout_advantages)))
        advantages = torch.reshape(torch.FloatTensor(advantages).to(self.config.device), (self.config.rollouts // self.config.batches * self.config.steps,))
        return advantages
