from typing import List

import numpy as np

from function_approximation import FunctionApproximation


class Approximation(FunctionApproximation):
    def eval(self, features, params) -> float:
        features = np.array(self._calc_features(features))
        return features @ np.array(params).transpose()

    def derivative_eval(self, features, params) -> List[float]:
        return self._calc_features(features)

    @staticmethod
    def _calc_features(features):
        n = len(features)
        for i in range(n):
            if features[i] != 0:
                if i == n - 2:
                    features[-1] = 2
                else:
                    features[-1] = 1
                    features[i] *= 2
                break

        return features
