import random
from typing import List, Tuple
from environment import Environment

ROAD_SYMBOL = x = "road"
STARTING_POINT_SYMBOL = s = "starting points"
FINISH_LINE_SYMBOL = e = "finish line"
BORDER_SYMBOL = 0

racetrack_turn = [
    [0, 0, 0, x, x, x, x, x, x, x, x, x, x, x, x, x, e],
    [0, 0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, e],
    [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, e],
    [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, e],
    [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, e],
    [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, e],
    [x, x, x, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0],
    [x, x, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0],
    [x, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [x, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, x, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, x, x, x, x, x, x, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, s, s, s, s, s, s, 0, 0, 0, 0, 0, 0, 0],
]


class RacingTrackState:
    def __init__(self, position: Tuple[int, int], velocity: Tuple[int, int]):
        self.position = position
        self.velocity = velocity


class RacingTrack(Environment):

    def __init__(self):
        self.initial_states = set()
        self.terminal_states = set()
        self.road = set()

        self.rows = len(racetrack_turn)
        self.columns = len(racetrack_turn[0])

        for i in range(self.rows):
            for j in range(self.columns):
                cell = racetrack_turn[i][j]
                if cell == ROAD_SYMBOL:
                    self.road.add((i, j))
                elif cell == STARTING_POINT_SYMBOL:
                    self.initial_states.add(RacingTrackState(position=(i, j), velocity=(0, 0)))
                elif cell == FINISH_LINE_SYMBOL:
                    self.terminal_states.add(RacingTrackState(position=(i, j), velocity=(0, 0)))

        self.actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    def get_initial_states(self) -> list:
        return self.initial_states

    def get_goal_states(self) -> list:
        return self.terminal_states

    def take_action(self, current_state: RacingTrackState, action) -> (object, int):
        if racetrack_turn[current_state.position[0]][current_state.position[1]] == FINISH_LINE_SYMBOL:
            return current_state, 0

        new_velocity = (current_state.velocity[0] + action[0], current_state.velocity[1] + action[1])
        new_position = (current_state.position[0] - new_velocity[0], current_state.position[1] + new_velocity[1])
        new_state = RacingTrackState(position=new_position, velocity=new_velocity)
        n = new_state.position[0]
        m = new_state.velocity[1]

        reward = -1
        if n < 0 or n >= self.rows or m < 0 or m >= self.columns or racetrack_turn[n][m] == BORDER_SYMBOL:
            new_state = random.choice(self.get_initial_states())
        elif racetrack_turn[n][m] == e:
            reward = 0

        return new_state, reward

    def get_actions(self, state: RacingTrackState) -> list:
        all_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

        valid_actions = []
        for action in all_actions:
            if state.velocity == (0, 0) and action == (0, 0):
                continue

            new_velocity = (state.velocity[0] + action[0], state.velocity[1] + action[1])
            if 0 <= new_velocity[0] <= 5 and 0 <= new_velocity[1] <= 5:
                valid_actions.append(action)
        return valid_actions

    def get_board(self) -> List[List[int]]:
        return racetrack_turn
