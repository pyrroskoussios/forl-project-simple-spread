import torch
import numpy as np

from agents.networks.rnn import RNN

class Disruptor:
    def __init__(self, config, observation_space, action_space):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space

        self.actor = RNN(observation_space + 1, self.config.num_nodes, self.config.num_hidden_layers, self.config.frame_stack * 2 * (self.config.num_agents - 1), output_splits=2).to(self.config.device)
        self.critic = RNN(observation_space + 1, self.config.num_nodes, self.config.num_hidden_layers, 1).to(self.config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_learning_rate)
        
    def chooseDisruption(self, disruptor_observation, exploit=False):
        with torch.no_grad():
            mean, log_sigma = self.actor(disruptor_observation)
        log_sigma = torch.clamp(log_sigma, min=-10, max=2)
        distribution = torch.distributions.Normal(mean, log_sigma.exp())
        if exploit:
            action = mean
        else:
            action = distribution.rsample()
        log_prob = (distribution.log_prob(action) - torch.log(2 * (1 - torch.tanh(action).pow(2)) + 1e-6)).sum()
        disruption = torch.tanh(action) * 2
        return action, log_prob, disruption

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

                mean, log_sigma = self.actor(rollout_data_batch["states"])
                log_sigma = torch.clamp(log_sigma, min=-10, max=2)
                distribution = torch.distributions.Normal(mean, log_sigma.exp())
                current_probability = (distribution.log_prob(rollout_data_batch["actions"]) - torch.log(2 * (1 - torch.tanh(rollout_data_batch["actions"]).pow(2)) + 1e-6)).sum(axis=1)
                log_probability_ratio = torch.exp(current_probability - rollout_data_batch["probabilities"])
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
