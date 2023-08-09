import numpy as np

from policy import Policy
from environment import Environment
from function_approximation import FunctionApproximation


class GreedyPolicy(Policy):

    def __init__(self, env: Environment, function_approximation: FunctionApproximation):
        self.env = env
        self.function_approximation = function_approximation
        self.weights = []

    def get_action(self, state, weights=None):
        return self._get_action_with_highest_value(state, self.env.get_actions(state), weights)

    def _get_action_with_highest_value(self, state, actions: list, weights):
        best_action = None
        max_value = float('-inf')
        for action in actions:
            features_set = self.env.get_features_set(state, action)
            value = self.function_approximation.eval(features=features_set, params=weights)
            if value > max_value:
                max_value = value
                best_action = action

        return best_action
