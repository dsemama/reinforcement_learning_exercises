from environment import Environment
from table import Table


class TargetFunctions:
    @staticmethod
    def incremental_average_step_towards_target_value(value,
                                                      step_size,
                                                      reward,
                                                      discounting_factor,
                                                      target_value):
        return value + step_size * (reward + discounting_factor * target_value - value)

    @staticmethod
    def max_state_action_value(env: Environment, q: Table, default_value: float, state):
        actions = env.get_actions(state)
        max_value = float('-inf')
        for action in actions:
            max_value = max(max_value, q.get(state, action, default_value))

        return max_value
