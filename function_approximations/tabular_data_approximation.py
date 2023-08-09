from table import Table
from function_approximation import FunctionApproximation


class TabularDataApproximation(FunctionApproximation):
    def __init__(self, q_table: Table):
        self.q_table = q_table

    def eval(self, features, _params) -> float:
        self.q_table.get_key(tuple(features))

    def derivative_eval(self, features, params) -> float:
        return 1
