import math
from deprecated import deprecated


class Table:
    def __init__(self, timestamp_threshold=float('inf'), bonus_factor: float = 1):
        self.q = {}
        self._updated_time = {}
        self.current_timestamp = 0
        self.timestamp_threshold = timestamp_threshold
        self.bonus_factor = bonus_factor

    @deprecated
    def get(self, state, action, default=None, enable_bonus_reward_for_old_values=True):
        return self.get_key(
            key=(state, action),
            default=default,
            enable_bonus_reward_for_old_values=enable_bonus_reward_for_old_values,
        )

    @deprecated
    def set(self, state, action, value):
        self.set_key(key=(state, action), value=value)

    def get_key(self, key, default=None, enable_bonus_reward_for_old_values=True):
        if key not in self.q:
            self.set_key(key, default)

        value = self.q[key]
        t = self.current_timestamp - self._updated_time[key]
        if enable_bonus_reward_for_old_values and t > self.timestamp_threshold:
            value += self.bonus_factor * math.sqrt(t)

        return self.q[key]

    def set_key(self, key, value):
        self.q[key] = value
        self._updated_time[key] = self.current_timestamp
        self.current_timestamp += 1

    def get_keys(self):
        return self.q.keys()

    def _updated_lately(self, key) -> bool:
        return self.current_timestamp - self._updated_time[key] <= self.timestamp_threshold
