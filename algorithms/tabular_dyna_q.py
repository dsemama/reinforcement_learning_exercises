from typing import List

import numpy as np

from algorithm import Algorithm
from policy import Policy
from model import Model
from environment import Environment
from algorithms.target_functions import TargetFunctions
from table import Table
import random


class TabularDynaQ(Algorithm):
    def __init__(self, env: Environment, policy: Policy, model: Model, step_size: float, episodes: int,
                 planning_steps: int, discounting_factor: float, q_table):
        self.env = env
        self.policy = policy
        self.model = model
        self.step_size = step_size
        self.episodes = episodes
        self.planning_steps = planning_steps
        self.discounting_factor = discounting_factor
        self.q = q_table

    def execute(self, debug=False) -> List[float]:
        episode = 1
        rewards_over_episodes = [0]
        state = random.choice(self.env.get_initial_states())
        while episode <= self.episodes:
            cumulative_reward = 0

            if debug and episode % 100 == 0:
                print("episode no. {}".format(episode))

            action = self.policy.get_action(state)

            next_state, reward = self.env.take_action(state, action)
            cumulative_reward += reward
            self.q.set(state, action, self._get_next_step_q_value(state, action, next_state, reward))
            self.model.set(state, action, next_state, reward)

            for i in range(self.planning_steps):
                state, action = self.model.get_visited_state_action_pair()
                next_state, reward = self.model.get(state, action)
                cumulative_reward += reward
                self.q.set(state, action, self._get_next_step_q_value(state, action, next_state, reward))

            state = next_state
            episode += 1
            rewards_over_episodes.append(rewards_over_episodes[-1] + cumulative_reward)

        return rewards_over_episodes

    def _get_next_step_q_value(self, state, action, next_state, reward):
        target_value = TargetFunctions.max_state_action_value(self.env, self.q, 0, next_state)
        q_value = self.q.get(state, action, 0)
        return TargetFunctions.incremental_average_step_towards_target_value(q_value, self.step_size, reward,
                                                                             self.discounting_factor, target_value)


class DeterministicModel(Model):
    def __init__(self):
        self.visited = set()
        self.table = Table()

    def set(self, state, action, next_state, reward):
        self.visited.add((state, action))
        self.table.set(state, action, (next_state, reward))

    def get(self, state, action) -> (object, int):
        next_state, reward = self.table.get(state, action)
        return next_state, reward

    def get_visited_state_action_pair(self):
        return random.choice(list(self.visited))
