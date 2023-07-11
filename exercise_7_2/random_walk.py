import math
from typing import List
from environment import Environment


class RandomWalk(Environment):

    def __init__(self, states: int, left_terminal_state_reward: int, right_terminal_state_reward: int):
        assert states % 2 == 1

        self.n = states
        self.center = math.floor(states / 2) + 1
        self.actions = ['left', 'right']
        self.left_terminal_state = 0
        self.right_terminal_state = self.n + 1
        self.left_terminal_state_reward = left_terminal_state_reward
        self.right_terminal_state_reward = right_terminal_state_reward

    def get_initial_states(self) -> list:
        return [self.center]

    def get_goal_states(self) -> list:
        return [self.left_terminal_state, self.right_terminal_state]

    def take_action(self, current_state, action) -> (object, int):
        next_state = current_state + 1 if action == 'right' else current_state - 1
        reward = 0
        if next_state == self.left_terminal_state:
            reward = self.left_terminal_state_reward

        if next_state == self.right_terminal_state:
            reward = self.right_terminal_state_reward

        return next_state, reward

    def get_actions(self, state) -> list:
        return self.actions

    def get_board(self) -> List[List[int]]:
        assert False, "not supported"
