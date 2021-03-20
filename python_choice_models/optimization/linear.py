# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from python_choice_models.settings import Settings
from python_choice_models.utils import time_for_optimization
import cplex


class LinearProblem(object):
    def amount_of_variables(self):
        raise NotImplementedError('Subclass responsibility')

    def objective_coefficients(self):
        raise NotImplementedError('Subclass responsibility')

    def lower_bounds(self):
        raise NotImplementedError('Subclass responsibility')

    def upper_bounds(self):
        raise NotImplementedError('Subclass responsibility')

    def variable_types(self):
        raise NotImplementedError('Subclass responsibility')

    def variable_names(self):
        raise NotImplementedError('Subclass responsibility')

    def constraints(self):
        raise NotImplementedError('Subclass responsibility')


class Constraints(object):
    def constraints(self):
        raise NotImplementedError('Subclass responsibility')


class LinearSolver(object):
    def solve(self, linear_problem, profiler):
        problem = cplex.Cplex()

        problem.parameters.timelimit.set(self.cpu_time(profiler))

        problem.set_log_stream(None)
        problem.set_error_stream(None)
        problem.set_warning_stream(None)
        problem.set_results_stream(None)

        problem.objective.set_sense(problem.objective.sense.maximize)

        problem.variables.add(obj=linear_problem.objective_coefficients(),
                              lb=linear_problem.lower_bounds(),
                              ub=linear_problem.upper_bounds(),
                              types=linear_problem.variable_types(),
                              names=linear_problem.variable_names())

        problem.linear_constraints.add(lin_expr=linear_problem.constraints()['linear_expressions'],
                                       senses=''.join(linear_problem.constraints()['senses']),
                                       rhs=linear_problem.constraints()['independent_terms'],
                                       names=linear_problem.constraints()['names'])

        problem.solve()

        print('')
        print(('MIP Finished: %s' % problem.solution.get_status_string()))
        print('')

        amount_solutions = 3
        solution_pool = problem.solution.pool

        all_solutions = []
        for solution_number in range(solution_pool.get_num()):
            all_solutions.append((solution_pool.get_objective_value(solution_number), solution_number))

        final_solutions = []
        for solution_number in [x[1] for x in sorted(all_solutions, key=lambda y: y[0])[-amount_solutions:]]:
            values = {v: solution_pool.get_values(solution_number, v) for v in problem.variables.get_names()}
            final_solutions.append((problem.solution.pool.get_objective_value(solution_number), values))

        return final_solutions

    def cpu_time(self, profiler):
        return time_for_optimization(partial_time=Settings.instance().linear_solver_partial_time_limit(),
                                     total_time=Settings.instance().solver_total_time_limit(),
                                     profiler=profiler)

