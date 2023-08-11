import random
from typing import List

import numpy as np

from environment import Environment


class BairdsCounterexampleEnvironment(Environment):
    def __init__(self):
        self.actions = ['solid', 'dashed']
        self.states = 7

    def get_initial_states(self) -> list:
        return list(range(1, self.states + 1))

    def get_goal_states(self) -> list:
        return []

    def take_action(self, current_state, action) -> (object, int):
        if action == 1:
            return self.states

        return random.randint(1, self.states - 1), 0

    def get_actions(self, state) -> list:
        return ['solid', 'dashed']

    def get_board(self) -> List[List[int]]:
        raise NotImplemented("Not supported")

    def get_features_set(self, state, _action):
        features = list(np.zeros(self.states+1))
        features[state-1] = 1
        return features
