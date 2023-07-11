import random

x = "road"
s = "starting points"
e = "finish line"
# 0 - border
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

starting_points = set()
finish_line = set()
road = set()

rows = len(racetrack_turn)
columns = len(racetrack_turn[0])

for i in range(rows):
    for j in range(columns):
        cell = racetrack_turn[i][j]
        if cell == x:
            road.add((i, j))
        elif cell == s:
            starting_points.add((i, j))
        elif cell == e:
            finish_line.add((i, j))


def get_possible_actions(state):
    velocity = state[1]
    all_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    valid_actions = []
    for action in all_actions:
        if velocity == (0, 0) == action:
            continue

        new_velocity = (velocity[0] + action[0], velocity[1] + action[1])
        if 0 <= new_velocity[0] <= 5 and 0 <= new_velocity[1] <= 5:
            valid_actions.append(action)
    return valid_actions


def get_starting_state():
    return random.choice(tuple(starting_points)), (0, 0)


def get_reward(state, action):
    current_position = (state[0][0], state[0][1])
    if racetrack_turn[current_position[0]][current_position[1]] == e:
        return state, 0

    new_velocity = (state[1][0] + action[0], state[1][1] + action[1])
    new_position = (current_position[0] - new_velocity[0], current_position[1] + new_velocity[1])
    n = new_position[0]
    m = new_position[1]

    reward = -1
    if n < 0 or n >= rows or m < 0 or m >= columns or racetrack_turn[n][m] == 0:
        (new_position, new_velocity) = get_starting_state()
    elif racetrack_turn[n][m] == e:
        reward = 0

    state = (new_position, new_velocity)

    return state, reward
