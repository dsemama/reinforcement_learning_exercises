from typing import List

from state import State


class WindyGridworldState(State):
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def encode(self) -> List[int]:
        return [self.i, self.j]

    @staticmethod
    def decode(decoded_state: List[int]) -> 'WindyGridworldState':
        return WindyGridworldState(decoded_state[0], decoded_state[1])

    def get_features_dimension(self) -> int:
        return 2

    def __eq__(self, other: 'WindyGridworldState'):
        if other is None:
            return False
        return self.encode() == other.encode()
