import random
from typing import List

from environment import Environment

_board = [
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [-1, 0, 0, 1, 1, 1, 2, -2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
]

actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class WindyGridworld(Environment):

    def __init__(self, enable_stochastic_wind=False):
        self.initial_states = []
        self.goal_states = []
        self.board = _board
        self.n = len(self.board)
        self.m = len(self.board[0])
        self.enable_stochastic_wind = enable_stochastic_wind

        for i in range(self.n):
            for j in range(self.m):
                if self.board[i][j] == -1:
                    self.initial_states.append((i, j))
                elif self.board[i][j] == -2:
                    self.goal_states.append((i, j))

        self.actions = actions

    def get_initial_states(self) -> list:
        return self.initial_states

    def get_goal_states(self) -> list:
        return self.goal_states

    def take_action(self, current_state, action):
        i, j = current_state
        if self._is_inside_the_board(current_state) and self.board[i][j] in self.goal_states:
            return current_state, 0

        return self._move(current_state, action), -1

    def get_actions(self, state) -> list:
        return self.actions

    def get_board(self) -> List[List[int]]:
        return self.board

    def _move(self, state, action):
        wind = self._get_wind(state)
        i, j = state[0] + action[0] - wind, state[1] + action[1]

        i = min(max(i, 0), self.n - 1)
        j = min(max(j, 0), self.m - 1)
        return i, j

    def _get_wind(self, state):
        i, j = state
        wind = max(0, self.board[i][j])

        if wind != 0 and self.enable_stochastic_wind:
            wind = random.randint(0, wind)

        return wind

    def _is_inside_the_board(self, state):
        i, j = state
        return 0 <= i <= self.n - 1 and 0 <= j <= self.m - 1
