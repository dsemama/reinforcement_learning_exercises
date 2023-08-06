import random
from typing import List

from environment import Environment
from environments.simple_maze.simple_maze_state import SimpleMazeState

board = [
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', 'G'],
    [' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' '],
    [' ', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' '],
    ['S', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
]


class SimpleMaze(Environment):
    def __init__(self):
        self.initial_states = []
        self.goal_states = []
        self.n = len(board)
        self.m = len(board[0])
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(self.n):
            for j in range(self.m):
                if board[i][j] == 'S':
                    self.initial_states.append(SimpleMazeState(i, j))
                elif board[i][j] == 'G':
                    self.goal_states.append(SimpleMazeState(i, j))

    def get_initial_states(self) -> list:
        return self.initial_states

    def get_goal_states(self) -> list:
        return self.goal_states

    def take_action(self, current_state: SimpleMazeState, action) -> (object, int):
        if current_state in self.goal_states:
            return random.choice(self.initial_states), 0

        assert self._is_valid_action(current_state, action)

        next_state = self._next_state(current_state, action)
        reward = 1 if next_state in self.goal_states else 0
        return next_state, reward

    def get_actions(self, state: SimpleMazeState) -> list:
        possible_actions = []
        for action in self.actions:
            if self._is_valid_action(state, action):
                possible_actions.append(action)

        return possible_actions

    def _is_valid_action(self, state: SimpleMazeState, action):
        i, j = self._next_state(state, action)
        return 0 <= i < self.n and 0 <= j < self.m and board[i][j] != 'X'

    def get_board(self) -> List[List[int]]:
        return board

    @staticmethod
    def _next_state(state: SimpleMazeState, action):
        return state.i + action[0], state.j + action[1]
