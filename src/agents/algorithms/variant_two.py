import torch

from agents.networks.mlp import MLP

class VariantTwo:
    def __init__(self, config, observation_space, action_space):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space

        self.actor = MLP(observation_space, self.config.num_nodes, self.config.num_hidden_layers, action_space, output_activation="softmax").to(self.config.device)
        self.critic = MLP(observation_space, self.config.num_nodes, self.config.num_hidden_layers, 1).to(self.config.device)
        self.global_reward_estimator = MLP(observation_space + action_space, 24, 0, 1).to(self.config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_learning_rate)
        self.global_reward_estimator_optimizer = torch.optim.Adam(self.global_reward_estimator.parameters(), lr=self.config.critic_learning_rate)

        self.resetEpisodicValues()
        
    def chooseAction(self, state, exploit=False):
        with torch.no_grad():
            probability = self.actor(state)
        if exploit:
            action = probability.argmax().item()
        else:
            action = torch.multinomial(probability, 1).item()
        return action, probability 

    def getCriticWeights(self):
        return self.critic.state_dict()
    
    def getGlobalRewardEstimatorWeights(self):
        return self.global_reward_estimator.state_dict()

    def getLongTermReward(self):
        return self.long_term_reward

    def setCriticWeights(self, critic_weights):
        self.critic.load_state_dict(critic_weights)
    
    def setGlobalRewardEstimatorWeights(self, global_reward_estimator_weights):
        self.global_reward_estimator.load_state_dict(global_reward_estimator_weights)
    
    def setLongTermReward(self, long_term_reward):
        self.long_term_reward = long_term_reward

    def resetEpisodicValues(self):
        self.long_term_reward = 0

    def update(self, state, action, next_state, reward):
        global_reward = self.global_reward_estimator(torch.cat((state, self._oneHotAction(action))))
        global_reward_estimator_loss = (reward - global_reward).pow(2) 

        state_value = self.critic(state)
        next_state_value = self.critic(next_state)
        td_error = (reward - self.long_term_reward) + (next_state_value - state_value)
        critic_loss = td_error.pow(2)

        probability = self.actor(state)
        log_probability = probability[action].log()
        with torch.no_grad():
            global_td_error = (global_reward - self.long_term_reward) + (next_state_value - state_value)
        actor_loss = -log_probability * global_td_error

        self.long_term_reward = (1 - self.config.critic_learning_rate) * self.long_term_reward + self.config.critic_learning_rate * reward

        self.global_reward_estimator_optimizer.zero_grad()
        global_reward_estimator_loss.backward()
        self.global_reward_estimator_optimizer.step()
        global_reward_estimator_loss = global_reward_estimator_loss.detach().cpu().item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        critic_loss = critic_loss.detach().cpu().item()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        actor_loss = actor_loss.detach().cpu().item()
        
        loss = (actor_loss, critic_loss, global_reward_estimator_loss)
        return loss

    def _oneHotAction(self, action):
        one_hot_action = torch.zeros(self.action_space).to(self.config.device)
        one_hot_action[action] = 1
        return one_hot_action
