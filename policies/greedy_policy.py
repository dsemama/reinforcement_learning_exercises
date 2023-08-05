import random
from policy import Policy
from environment import Environment
from table import Table


class GreedyPolicy(Policy):

    def __init__(self, env: Environment, q_table: Table):
        self.env = env
        self.q_table = q_table

    def get_action(self, state):
        return self._get_action_with_highest_value(state, self.env.get_actions(state))

    def _get_action_with_highest_value(self, state, actions: list):
        best_action = None
        max_value = float('-inf')
        for action in actions:
            value = self.q_table.get(state, action, 0)
            if value > max_value:
                max_value = value
                best_action = action

        return best_action