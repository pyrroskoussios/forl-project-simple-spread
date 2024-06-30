# noqa: D212, D415
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v3` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |


This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
import random
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def generate_colors(self, N):
        # generates N distinct colors
        colors = []
        for _ in range(N):
            color = np.random.rand(3)
            colors.append(color)
        return colors

    def make_world(self, N=3):
        # pyrros edit
        self.communication_reach = 3
        self.curriculum_learning = False
        self.centralized = False
        # hello to anyone reading :D

        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        colors = self.generate_colors(N)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            #agent.color = colors[i]

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            #landmark.color = colors[i]
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        
        if agent.collide:
            for a in world.agents:
                rew -= 1.0 * (self.is_collision(a, agent) and a != agent)

        """
        if any(agent.action.u):
            rew -= 0.01
        print("--")
        print(agent.state.last_p_pos)
        print(agent.state.p_pos)
        """

        return rew

    def global_reward(self, world):
        """
        agents_to_remove = []
        for agent1 in world.agents:
            for agent2 in world.agents:
                if agent1 is not agent2 and self.is_collision(agent1, agent2):
                    agents_to_remove.extend([agent1, agent2])
        world.agents = [x for x in world.agents if x not in agents_to_remove]
        """

        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        """
        for agent in world.agents:
            agent_id = int(agent.name.split("_")[1])
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[agent_id].state.p_pos)))
            rew -= dist
        """
        return rew

    def observation(self, agent, world):
        if self.curriculum_learning:
            sorted_entities = []
            for id, entity in enumerate(world.landmarks):  # world.entities:
                distance = np.sqrt(np.sum(np.power(entity.state.p_pos - agent.state.p_pos, 2)))
                sorted_entities.append((distance, id, entity))
            sorted_entities.sort()

            sorted_agents = []
            for id, other in enumerate(world.agents):  # world.entities:
                if other is agent:
                    continue
                distance = np.sqrt(np.sum(np.power(other.state.p_pos - agent.state.p_pos, 2)))
                sorted_agents.append((distance, id, other))
            sorted_agents.sort()
            
            reachable_entities = sorted_entities[:self.communication_reach]
            reachable_agents = sorted_agents[:self.communication_reach - 1]
            #landmark_ids = np.array([x[1] * (1 / (self.N - 1)) for x in reachable_entities])
            landmark_delta = [x[2].state.p_pos - agent.state.p_pos for x in reachable_entities]
            #agent_ids = np.array([x[1] * (1 / (self.N - 1)) for x in reachable_agents])
            agent_delta_pos = [x[2].state.p_pos - agent.state.p_pos for x in reachable_agents]
            #comm = [x[2].state.c for x in reachable_agents]
            observation = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmark_delta + agent_delta_pos)
        else:
            entity_pos = []
            for entity in world.landmarks:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            
            #comm = []
            other_pos = []
            for other in world.agents:
                if other is agent:
                    continue
                #comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)

            observation = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

        return observation
