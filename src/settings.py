NLP_LOWER_BOUND_INF = -1e19
NLP_UPPER_BOUND_INF = 1e19


class Settings(object):
    _instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            raise Exception('Must set settings for a specific estimator')
        return cls._instance

    @classmethod
    def new(cls, linear_solver_partial_time_limit,
            non_linear_solver_partial_time_limit, solver_total_time_limit):
        cls._instance = cls(linear_solver_partial_time_limit=linear_solver_partial_time_limit,
                            non_linear_solver_partial_time_limit=non_linear_solver_partial_time_limit,
                            solver_total_time_limit=solver_total_time_limit)

    def __init__(self, linear_solver_partial_time_limit,
                 non_linear_solver_partial_time_limit, solver_total_time_limit):
        self._linear_solver_partial_time_limit = linear_solver_partial_time_limit
        self._non_linear_solver_partial_time_limit = non_linear_solver_partial_time_limit
        self._solver_total_time_limit = solver_total_time_limit

    def linear_solver_partial_time_limit(self):
        return self._linear_solver_partial_time_limit

    def non_linear_solver_partial_time_limit(self):
        return self._non_linear_solver_partial_time_limit

    def solver_total_time_limit(self):
        return self._solver_total_time_limit
