from typing import List, Tuple

from state import State


class SimpleMazeState(State):
    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j

    def encode(self) -> Tuple[int]:
        return self.i, self.j

    @staticmethod
    def decode(decoded_state: Tuple[int]) -> 'SimpleMazeState':
        return SimpleMazeState(decoded_state[0], decoded_state[1])

    def get_features_dimension(self) -> int:
        return 2

    def __eq__(self, other):
        if isinstance(other, SimpleMazeState):
            return self.encode() == other.encode()
        return False

    def __hash__(self):
        return hash(self.encode())
