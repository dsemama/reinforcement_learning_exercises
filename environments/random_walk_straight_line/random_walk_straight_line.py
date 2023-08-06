import math
from typing import List
from environment import Environment
from environments.random_walk_straight_line.random_walk_straight_line_state import RandomWalkStraightLineState


class RandomWalkStraightLine(Environment):

    def __init__(self, states_count: int, left_terminal_state_reward: int, right_terminal_state_reward: int):
        assert states_count % 2 == 1

        self.n = states_count
        self.center = RandomWalkStraightLineState(math.floor(states_count / 2) + 1)
        self.actions = ['left', 'right']
        self.left_terminal_state = RandomWalkStraightLineState(0)
        self.right_terminal_state = RandomWalkStraightLineState(self.n + 1)
        self.left_terminal_state_reward = left_terminal_state_reward
        self.right_terminal_state_reward = right_terminal_state_reward

    def get_initial_states(self) -> list:
        return [self.center]

    def get_goal_states(self) -> list:
        return [self.left_terminal_state, self.right_terminal_state]

    def take_action(self, current_state: 'RandomWalkStraightLineState', action) -> (object, int):
        next_state = RandomWalkStraightLineState(current_state.position + 1) \
            if action == 'right' \
            else RandomWalkStraightLineState(current_state.position - 1)

        reward = 0
        if next_state.position == self.left_terminal_state.position:
            reward = self.left_terminal_state_reward

        if next_state.position == self.right_terminal_state.position:
            reward = self.right_terminal_state_reward

        return next_state, reward

    def get_actions(self, state: 'RandomWalkStraightLineState') -> list:
        return self.actions

    def get_board(self) -> List[List[int]]:
        assert False, "not supported"
