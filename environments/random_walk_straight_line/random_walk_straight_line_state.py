from typing import Tuple, List

from state import State


class RandomWalkStraightLineState(State):
    def __init__(self, position: int):
        self.position = position

    def encode(self) -> List[int]:
        return [self.position]

    @staticmethod
    def decode(decoded_state: List[int]) -> 'RandomWalkStraightLineState':
        assert len(decoded_state) == 1, "RandomWalkStraightLineState decode representation is a single integer"
        return RandomWalkStraightLineState(decoded_state[0])

    def get_features_dimension(self) -> int:
        return 1

    def __eq__(self, other):
        if isinstance(other, RandomWalkStraightLineState):
            return self.encode() == other.encode()
        return False
