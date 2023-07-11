import math


class Table:
    def __init__(self, timestamp_threshold=float('inf'), bonus_factor: float = 1):
        self.q = {}
        self._updated_time = {}
        self.current_timestamp = 0
        self.timestamp_threshold = timestamp_threshold
        self.bonus_factor = bonus_factor

    def get_keys(self):
        return self.q.keys()

    def get(self, state, action, default=None, enable_bonus_reward_for_old_values=True):
        key = (state, action)
        if key not in self.q:
            self.set(state, action, default)

        value = self.q[key]
        t = self.current_timestamp - self._updated_time[key]
        if enable_bonus_reward_for_old_values and t > self.timestamp_threshold:
            value += self.bonus_factor * math.sqrt(t)

        return self.q[key]

    def _updated_lately(self, key) -> bool:
        return self.current_timestamp - self._updated_time[key] <= self.timestamp_threshold

    def set(self, state, action, value):
        self.q[(state, action)] = value

        # not thread safe
        self._updated_time[(state, action)] = self.current_timestamp
        self.current_timestamp += 1
