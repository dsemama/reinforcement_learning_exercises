import random
from policy import Policy
from environment import Environment
from table import Table


class EpsilonGreedyPolicy(Policy):

    def __init__(self, env: Environment, q_table: Table, exploration_rate: int):
        self.exploration_rate = exploration_rate
        self.env = env
        self.q_table = q_table

    def get_action(self, state):
        actions = self.env.get_actions(state)
        if random.random() <= self.exploration_rate:
            return random.choice(actions)

        best_action = None
        max_value = float('-inf')
        for action in actions:
            value = self.q_table.get(state, action, 0)
            if value > max_value:
                max_value = value
                best_action = action

        return best_action
