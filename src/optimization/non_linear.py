# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from numpy import array
from settings import Settings
from utils import finite_difference, time_for_optimization
import time
from scipy.optimize import minimize


class NonLinearSolver(object):
    @classmethod
    def default(cls):
        return ScipySolver()

    def solve(self, non_linear_problem, profiler):
        raise NotImplemented('Subclass responsibility')

    def cpu_time(self, profiler):
        return time_for_optimization(partial_time=Settings.instance().non_linear_solver_partial_time_limit(),
                                     total_time=Settings.instance().solver_total_time_limit(),
                                     profiler=profiler)


class TookTooLong(Exception):
    def __init__(self, objective_value, parameters):
        self.objective_value = objective_value
        self.parameters = parameters


class FailToOptimize(Exception):
    def __init__(self, reason):
        self.reason = reason


class ScipySolver(NonLinearSolver):
    def bounds_for(self, non_linear_problem):
        lower = list(non_linear_problem.constraints().lower_bounds_vector())
        upper = list(non_linear_problem.constraints().upper_bounds_vector())
        return list(zip(lower, upper))

    def constraints_for(self, non_linear_problem):
        lower_c = list(non_linear_problem.constraints().lower_bounds_over_constraints_vector())
        upper_c = list(non_linear_problem.constraints().upper_bounds_over_constraints_vector())
        evaluator = non_linear_problem.constraints().constraints_evaluator()

        i = 0
        constraints = []
        for l, u in zip(lower_c, upper_c):
            if l == u:
                constraints.append({'type': 'eq', 'fun': (lambda j: lambda x: evaluator(x)[j] - l)(i)})
            else:
                constraints.append({'type': 'ineq', 'fun': (lambda j: lambda x: u - evaluator(x)[j])(i)})
                constraints.append({'type': 'ineq', 'fun': (lambda j: lambda x: evaluator(x)[j] - l)(i)})
            i += 1

        return constraints

    def solve(self, non_linear_problem, profiler):
        time_limit = self.cpu_time(profiler)
        start_time = time.time()

        def iteration_callback(x):
            objective = non_linear_problem.objective_function(x)
            profiler.stop_iteration(objective)
            profiler.start_iteration()
            if time.time() - start_time > time_limit:
                raise TookTooLong(objective, x)

        bounds = self.bounds_for(non_linear_problem)
        constraints = self.constraints_for(non_linear_problem)

        profiler.start_iteration()
        try:
            r = minimize(fun=non_linear_problem.objective_function, x0=array(non_linear_problem.initial_solution()),
                         jac=False, bounds=bounds, constraints=constraints, callback=iteration_callback,
                         method='SLSQP', options={'maxiter': 100000})
            fun = r.fun
            x = r.x
            success = r.success
            status = r.status
            message = r.message
        except TookTooLong as e:
            fun = e.objective_value
            x = e.parameters
            success = True
        profiler.stop_iteration(fun)

        if not success:
            raise FailToOptimize(reason='Falla al optimizar. Estado de terminacion de scipy %s. %s' % (status, message))

        return x


class NonLinearProblem(object):
    def initial_solution(self):
        raise NotImplementedError('Subclass responsibility')

    def objective_function(self, vector):
        raise NotImplementedError('Subclass responsibility')

    def jacobian(self, vector):
        # TODO: Is it bad to define this 'finite difference' function each time jacobian is called?
        return finite_difference(self.objective_function)(vector)

    def amount_of_variables(self):
        raise NotImplementedError('Subclass responsibility')

    def constraints(self):
        raise NotImplementedError('Subclass responsibility')


class Constraints(object):
    def lower_bounds_vector(self):
        """
            Lower bounds for parameters vector. Can be pyipopt.NLP_LOWER_BOUND_INF.
        """
        return array([])

    def upper_bounds_vector(self):
        """
            Upper bounds for parameters vector. Can be pyipopt.NLP_UPPER_BOUND_INF.
        """
        return array([])

    def amount_of_constraints(self):
        """
            Amount of constraints on model
        """
        return 0

    def lower_bounds_over_constraints_vector(self):
        """
            Lower bounds for each constraints. Can be pyipopt.NLP_LOWER_BOUND_INF.
        """
        return array([])

    def upper_bounds_over_constraints_vector(self):
        """
            Upper bounds for each constraints. Can be pyipopt.NLP_UPPER_BOUND_INF.
        """
        return array([])

    def non_zero_parameters_on_constraints_jacobian(self):
        """
            Non zero values on constraints jacobian matrix.
        """
        return 0

    def constraints_evaluator(self):
        """
            A function that evaluates constraints.
        """
        def evaluator(x):
            return 0.0
        return evaluator

    def constraints_jacobian_evaluator(self):
        """
            A function that evaluates constraints jacobian matrix.
        """
        def jacobian_evaluator(x, flag):
            if flag:
                return array([]), array([])
            else:
                return array([])
        return jacobian_evaluator
