import random
from policy import Policy


class BehaviorPolicy(Policy):
    def get_action(self, state, weights=None):
        pick = random.randint(1, 7)
        if pick == 1:
            return 'solid'
        return 'dashed'


class TargetPolicy(Policy):
    def get_action(self, state, weights=None):
        return 'solid'
