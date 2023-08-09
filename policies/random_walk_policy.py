import random
from policy import Policy
from environment import Environment


class RandomWalkPolicy(Policy):
    def __init__(self, env: Environment):
        self.env = env

    def get_action(self, state, weights=None):
        return random.choice(self.env.get_actions(state))
