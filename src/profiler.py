# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

# import matplotlib.pyplot as plt
import time


ACCEPTABLE_ITERATIONS = 5
ACCEPTABLE_OBJ_DIFFERENCE = 1e-6
BUDGET_TIME_LIMIT = 60 * 30


class ConvergenceCriteria(object):
    def would_stop_this(self, profiler):
        raise NotImplementedError('Subclass responsibility')

    def reset_for(self, profiler):
        pass


class ObjectiveValueCriteria(ConvergenceCriteria):
    def __init__(self, acceptable_iterations, acceptable_objective_difference):
        self._acceptable_iterations = acceptable_iterations
        self._acceptable_objective_difference = acceptable_objective_difference
        self._last_considered_iteration = 0

    def acceptable_iterations(self):
        return self._acceptable_iterations

    def acceptable_objective_difference(self):
        return self._acceptable_objective_difference

    def reset_for(self, profiler):
        self._last_considered_iteration = len(profiler.iterations())

    def would_stop_this(self, profiler):
        last_iterations = profiler.iterations()[self._last_considered_iteration:][-self.acceptable_iterations():]
        if len(last_iterations) == self.acceptable_iterations():
            differences = [abs(last_iterations[i].value() - last_iterations[i - 1].value()) for i in range(1, len(last_iterations))]
            return all([difference < self.acceptable_objective_difference() for difference in differences])
        return False


class TimeBudgetCriteria(ConvergenceCriteria):
    def __init__(self, time_limit):
        """
        time_limit: Time limit in seconds
        """
        self._time_limit = time_limit

    def time_limit(self):
        return self._time_limit

    def would_stop_this(self, profiler):
        return profiler.duration() > self.time_limit()


class MixedConvergenceCriteria(ConvergenceCriteria):
    def __init__(self, criteria):
        self._criteria = criteria

    def reset(self):
        for criteria in self._criteria:
            criteria.reset()

    def would_stop_this(self, profiler):
        return any([criteria.would_stop_this(profiler) for criteria in self._criteria])


class Iteration(object):
    def __init__(self):
        self._start_time = time.time()
        self._stop_time = None
        self._value = None

    def is_finished(self):
        return self._value is not None

    def finish_with(self, value):
        if self.is_finished():
            raise Exception('Finishing already finished iteration.')
        self._value = value
        self._stop_time = time.time()

    def value(self):
        return self._value

    def start_time(self):
        return self._start_time

    def stop_time(self):
        return self._stop_time

    def duration(self):
        return self.stop_time() - self.start_time()

    def as_json(self):
        return {'start': self.start_time(),
                'stop': self.stop_time(),
                'value': self.value()}

    def __repr__(self):
        data = (self.start_time(), self.stop_time(), self.duration(), self.value())
        return '< Start: %s ; Stop: %s ; Duration %s ; Value: %s >' % data


class Profiler(object):
    def __init__(self, verbose=True):
        self._verbose = verbose
        self._iterations = []
        time_criteria = TimeBudgetCriteria(BUDGET_TIME_LIMIT)
        objective_value_criteria = ObjectiveValueCriteria(ACCEPTABLE_ITERATIONS, ACCEPTABLE_OBJ_DIFFERENCE)
        self._convergence_criteria = MixedConvergenceCriteria(criteria=[time_criteria, objective_value_criteria])

    def iterations(self):
        return self._iterations

    def convergence_criteria(self):
        return self._convergence_criteria

    def json_iterations(self):
        return [i.as_json() for i in self.iterations()]

    def last_iteration(self):
        return self._iterations[-1]

    def first_iteration(self):
        return self._iterations[0]

    def start_iteration(self):
        self._iterations.append(Iteration())

    def stop_iteration(self, value):
        self.last_iteration().finish_with(value)
        self.show_progress()

    def show_progress(self):
        if self._verbose:
            if len(self.iterations()) % 10 == 1:
                print('----------------------')
                print('N#  \tTIME \tOBJ VALUE')
            print(('%s\t%ss\t%.8f' % (len(self.iterations()), int(self.duration()), self.last_iteration().value())))

    def duration(self):
        if len(self.iterations()) > 0:
            return self.last_iteration().stop_time() - self.first_iteration().start_time()
        return 0

    def should_stop(self):
        return self.convergence_criteria().would_stop_this(self)

    def reset_convergence_criteria(self):
        self.convergence_criteria().reset_for(self)

    def update_time(self):
        if len(self._iterations) > 2:
            self.start_iteration()
            self.stop_iteration(self._iterations[-2].value())