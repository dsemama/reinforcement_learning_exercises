from typing import List, Tuple

from state import State


class RacingTrackState(State):
    def __init__(self, position: Tuple[int, int], velocity: Tuple[int, int]):
        self.position = position
        self.velocity = velocity

    @staticmethod
    def decode(decoded_state) -> 'RacingTrackState':
        return RacingTrackState(
            position=(decoded_state[0], decoded_state[1]),
            velocity=(decoded_state[2], decoded_state[3]),
        )

    def get_features_dimension(self) -> int:
        return 4

    def encode(self) -> List[float]:
        return [self.position[0], self.position[1], self.velocity[0], self.velocity[1]]

    def __eq__(self, other):
        if isinstance(other, RacingTrackState):
            return self.encode() == other.encode()
        return False
