import os
import json
import argparse


class Config:
    def __init__(self, path=None):
        extras = self._parse_extras()
        
        if path:
            path = os.path.abspath(os.path.join(path, "config.json"))
        else:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "default_config.json"))

        config_file = self._parse_json(path)
        
        self.run_name = extras.run_name if extras.run_name else config_file["run_name"]
        self.environment = extras.environment if extras.environment else config_file["environment"]
        self.device = extras.device if extras.device else config_file["device"]
        self.algorithm = extras.algorithm if extras.algorithm else config_file["algorithm"]
        self.network_type = extras.network_type if extras.network_type else config_file["network_type"]
        self.iterative_disruptor_training = extras.iterative_disruptor_training if extras.iterative_disruptor_training is not None else config_file["iterative_disruptor_training"]
        self.curriculum_learning = extras.curriculum_learning if extras.curriculum_learning is not None else config_file["curriculum_learning"]
        self.centralized = extras.centralized if extras.centralized is not None else config_file["centralized"]
        self.log = extras.log if extras.log is not None else config_file["log"]
        self.num_agents = extras.num_agents if extras.num_agents else config_file["num_agents"]
        self.num_adversaries = extras.num_adversaries if extras.num_adversaries else config_file["num_adversaries"]
        self.iteration_length = extras.iteration_length if extras.iteration_length else config_file["iteration_length"]
        self.curriculum_stages = extras.curriculum_stages if extras.curriculum_stages else config_file["curriculum_stages"]
        self.disrupt_count = extras.disrupt_count if extras.disrupt_count else config_file["disrupt_count"]
        self.frame_stack = extras.frame_stack if extras.frame_stack else config_file["frame_stack"]
        self.communication_reach = extras.communication_reach if extras.communication_reach else config_file["communication_reach"]
        self.connectivity = extras.connectivity if extras.connectivity else config_file["connectivity"]
        self.episodes = extras.episodes if extras.episodes else config_file["episodes"]
        self.evaluation_episodes = extras.evaluation_episodes if extras.evaluation_episodes else config_file["evaluation_episodes"]
        self.evaluation_frequency = extras.evaluation_frequency if extras.evaluation_frequency else config_file["evaluation_frequency"]
        self.steps = extras.steps if extras.steps else config_file["steps"]
        self.rollouts = extras.rollouts if extras.rollouts else config_file["rollouts"]
        self.batches = extras.batches if extras.batches else config_file["batches"]
        self.epochs = extras.epochs if extras.epochs else config_file["epochs"]
        self.num_nodes = extras.num_nodes if extras.num_nodes else config_file["num_nodes"]
        self.num_hidden_layers = extras.num_hidden_layers if extras.num_hidden_layers else config_file["num_hidden_layers"]
        self.seed = extras.seed if extras.seed else config_file["seed"]
        self.discount_factor = extras.discount_factor if extras.discount_factor else config_file["discount_factor"]
        self.clip_value = extras.clip_value if extras.clip_value else config_file["clip_value"]
        self.actor_learning_rate = extras.actor_learning_rate if extras.actor_learning_rate else config_file["actor_learning_rate"]
        self.critic_learning_rate = extras.critic_learning_rate if extras.critic_learning_rate else config_file["critic_learning_rate"]

        self.is_adversarial_environment = True if self.environment == "simple_tag" else False
        self.is_rollout_algorithm = True if self.algorithm == "ppo" else False
        
        self._check_validity()
    
    def _parse_json(self, path):
        with open(path, "r") as file:
            config_file = json.load(file)
        return config_file

    def _parse_extras(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--run_name", type=str)
        parser.add_argument("--environment", type=str)
        parser.add_argument("--device", type=str)
        parser.add_argument("--algorithm", type=str)
        parser.add_argument("--network_type", type=str)
        parser.add_argument("--iterative_disruptor_training", type=str)
        parser.add_argument("--curriculum_learning", type=str)
        parser.add_argument("--centralized", type=str)
        parser.add_argument("--log", type=str)
        parser.add_argument("--communication_reach", type=int)
        parser.add_argument("--connectivity", type=int)
        parser.add_argument("--num_agents", type=int)
        parser.add_argument("--num_adversaries", type=int)
        parser.add_argument("--iteration_length", type=int)
        parser.add_argument("--curriculum_stages", type=int)
        parser.add_argument("--disrupt_count", type=int)
        parser.add_argument("--frame_stack", type=int)
        parser.add_argument("--episodes", type=int)
        parser.add_argument("--evaluation_episodes", type=int)
        parser.add_argument("--evaluation_frequency", type=int)
        parser.add_argument("--steps", type=int)
        parser.add_argument("--rollouts", type=int)
        parser.add_argument("--batches", type=int)
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--num_nodes", type=int)
        parser.add_argument("--num_hidden_layers", type=int)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--discount_factor", type=float)
        parser.add_argument("--clip_value", type=float)
        parser.add_argument("--actor_learning_rate", type=float)
        parser.add_argument("--critic_learning_rate", type=float)
        extras = parser.parse_args()
        
        bool_dict = {"True": True, "False": False}
        bool_args = ["log", "centralized", "iterative_disruptor_training", "curriculum_learning"]

        for bool_arg in bool_args:
            if getattr(extras, bool_arg):
                assert getattr(extras, bool_arg) in bool_dict, "Boolean value must be either 'True' or 'False'."
                setattr(extras, bool_arg, bool_dict[getattr(extras, bool_arg)])

        return extras

    def _check_validity(self):
        assert self.device in ["cpu", "cuda", "mps"], "Supported algorithms are: 'cpu', 'cuda', 'mps'."
        assert self.algorithm in ["variant_one", "variant_two", "ppo"], "Supported algorithms are: 'variant_one', 'variant_two', 'ppo'."
        assert self.network_type in ["mlp", "rnn"], "Supported network types are: 'mlp', 'rnn'."
        assert self.environment in ["simple", "simple_spread", "simple_tag", "simple_reference"], "Supported environments are: 'simple_spread', 'simple_tag'."
        assert self.connectivity > 1, "Connectivity must be larger than 1." 
        assert self.num_agents > 0, "Number of agents must be larger than 0." 
        assert self.episodes > 0, "Episodes must be larger than 0." 
        assert self.evaluation_episodes > 0, "Evaluation episodes must be larger than 0." 
        assert self.steps > 0, "Steps must be larger than 0." 
        assert self.num_nodes > 0, "Number of nodes must be larger than 0." 
        assert self.num_hidden_layers >= 0, "Number of hidden layers must be larger or equal to 0." 
        assert self.discount_factor >= 0 and self.discount_factor <= 1, "Discount factor must be larger than 0 and smaller than 1." 
        assert self.actor_learning_rate > 0, "Actor learning rate must be larger than 0." 
        assert self.critic_learning_rate > 0, "Critic learning rate must be larger than 0." 
        assert self.communication_reach > 0 and self.communication_reach <= self.num_agents, "Communication reach must be larger than zero and less than or equal to number of agents."
        if self.is_adversarial_environment:
            assert self.num_adversaries > 0, "Number of adversaries must be larger than 0." 
        if self.is_rollout_algorithm:
            assert self.rollouts > 0, "Rollouts must be larger than 0." 
            assert self.epochs > 0, "Epochs must be larger than 0." 
            assert self.batches > 0 and self.batches <= self.steps, "Batches must be larger than zero or less or equal to amount of steps."
            assert self.rollouts % self.batches == 0, "Batches must me a divisor of rollouts."
            assert self.episodes > self.rollouts, "Number of Episodes must be larger or equal to number of rollouts."
        else:
            assert self.num_agents >= self.connectivity, "Connectivity must be less or equal to number of agents."
        if self.network_type == "rnn":
            assert self.frame_stack > 0, "The RNN network type requires a frame stack larger or equal to 1."
        if self.network_type == "mlp":
            assert self.frame_stack == 1, "the MLP network type requires a frame stack of exactly 1."
        if self.iterative_disruptor_training:
            assert self.algorithm == "ppo", "Iterative disruptor training is only implemented for PPO."
            assert self.disrupt_count >= 1 and self.disrupt_count <= self.num_agents, "Disruption count must be value between 1 and number of agents."
        if self.curriculum_learning:
            assert self.is_rollout_algorithm, "Curriculum learning is only supported for PPO."
            assert self.log, "Curriculum learning requires logging to be turned on as it performs multiple of runs, loading models for the n+1 agents from the last run."
            assert self.curriculum_stages > 0, "Curriculum learning requires atleast 1 stage of adding agents."
            assert self.episodes > self.rollouts * self.evaluation_frequency, "Number of episodes must be larger than evaluation frequency multiplied by number of rollouts, as models need to be saved for the next stage to commence."

        



