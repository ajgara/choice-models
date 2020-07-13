from numpy import array
from math import log
from itertools import chain, combinations
from numba import jit
import numpy


ZERO_LOWER_BOUND = 1e-6
ONE_UPPER_BOUND = 1.0 - ZERO_LOWER_BOUND

FINITE_DIFFERENCE_DELTA = 1e-7


@jit
def safe_log_array(old_array):
    new_array = []
    for number in old_array:
        new_array.append(safe_log(number))
    return new_array


@jit
def safe_log(x):
    # This is to deal with infeasible optimization methods (those who don't care about evaluating objective function
    # inside constraints, this could cause evaluating outside log domain)
    if x > ZERO_LOWER_BOUND:
        return log(x)
    log_lower_bound = log(ZERO_LOWER_BOUND)
    a = 1 / (3 * ZERO_LOWER_BOUND * (3 * log_lower_bound * ZERO_LOWER_BOUND)**2)
    b = ZERO_LOWER_BOUND * (1 - 3 * log_lower_bound)
    return a * (x - b) ** 3


def finite_difference(function):
    def derivative(x):
        h = FINITE_DIFFERENCE_DELTA
        gradient = []
        x = list(x)
        for i, parameter in enumerate(x):
            plus = function(x[:i] + [parameter + h] + x[i + 1:])
            minus = function(x[:i] + [parameter - h] + x[i + 1:])
            gradient.append((plus - minus) / (2 * h))
        return array(gradient)
    return derivative


def generate_n_random_numbers_that_sum_one(n):
    distribution = [numpy.random.uniform(0, 1) for _ in range(n)]
    total = sum(distribution)

    for i in range(len(distribution)):
        distribution[i] = distribution[i] / total

    return distribution


def generate_n_equal_numbers_that_sum_one(n):
    head = [1.0 / n for _ in range(n - 1)]
    return head + [1.0 - sum(head)]


def generate_n_equal_numbers_that_sum_m(n, m):
    return [x * m for x in generate_n_equal_numbers_that_sum_one(n)]


def generate_n_random_numbers_that_sum_m(n, m):
    return [x * m for x in generate_n_random_numbers_that_sum_one(n)]


def time_for_optimization(partial_time, total_time, profiler):
    if not partial_time:
        return max(total_time - profiler.duration(), 0.01)
    return max(min(partial_time, total_time - profiler.duration()), 0.01)


def rindex(a_list, a_value):
    return len(a_list) - a_list[::-1].index(a_value) - 1


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
