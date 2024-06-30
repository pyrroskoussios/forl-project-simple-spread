import torch

from agents.networks.mlp import MLP

class VariantOne:
    def __init__(self, config, observation_space, action_space):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space

        self.actor = MLP(observation_space, self.config.num_nodes, self.config.num_hidden_layers, action_space, output_activation="softmax").to(self.config.device)
        self.critic = MLP(observation_space + action_space, self.config.num_nodes, self.config.num_hidden_layers, 1).to(self.config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_learning_rate)

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
    
    def setCriticWeights(self, critic_weights):
        self.critic.load_state_dict(critic_weights)

    def resetEpisodicValues(self):
        self.long_term_reward = 0

    def update(self, state, action, next_state, reward):
        next_action, _ = self.chooseAction(next_state)
        action_value = self.critic(torch.cat((state, self._oneHotAction(action))))
        next_action_value = self.critic(torch.cat((next_state, self._oneHotAction(next_action))))
        td_error = (reward - self.long_term_reward) + (next_action_value - action_value)
        critic_loss = td_error.pow(2)
        
        probability = self.actor(state)
        log_probability = probability[action].log()
        with torch.no_grad():
            average_other_action_value = 0
            for other_chosen_action in range(self.action_space):
                if other_chosen_action == action:
                    continue
                other_action_value = self.critic(torch.cat((state, self._oneHotAction(other_chosen_action))))
                average_other_action_value += probability[other_chosen_action] * other_action_value
            advantage = action_value - average_other_action_value 
        actor_loss = -log_probability * advantage

        self.long_term_reward = (1 - self.config.critic_learning_rate) * self.long_term_reward + self.config.critic_learning_rate * reward

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        critic_loss = critic_loss.detach().cpu().item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        actor_loss = actor_loss.detach().cpu().item()
        
        loss = (actor_loss, critic_loss)
        return loss

    def _oneHotAction(self, action):
        one_hot_action = torch.zeros(self.action_space).to(self.config.device)
        one_hot_action[action] = 1
        return one_hot_action
