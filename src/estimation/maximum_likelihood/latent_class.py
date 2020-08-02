# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

import time
from numpy import array, ones, zeros
from estimation import Estimator
from models import MultinomialLogitModel
from collections import defaultdict
from optimization.non_linear import NonLinearProblem, NonLinearSolver, Constraints
from settings import Settings
from utils import safe_log, ZERO_LOWER_BOUND, ONE_UPPER_BOUND, time_for_optimization


class LatentClassFrankWolfeEstimator(Estimator):
    def likelihood_loss_function_coefficients(self, transactions):
        sales_per_transaction = defaultdict(lambda: 0.0)
        for transaction in transactions:
            sales_per_transaction[transaction] += 1.0
        return [(transaction, amount_of_sales) for transaction, amount_of_sales in list(sales_per_transaction.items())]

    def look_for_new_mnl_model(self, model, likelihood_loss_function_coefficients):
        possible_mnl_model = MultinomialLogitModel.simple_deterministic(model.products)
        problem = NewMNLSubProblem(model, possible_mnl_model, likelihood_loss_function_coefficients)
        solution = NonLinearSolver.default().solve(problem, self.profiler())
        possible_mnl_model.update_parameters_from_vector(solution)
        return possible_mnl_model

    def update_weights_for(self, model, likelihood_loss_function_coefficients):
        problem = NewWeightsSubProblem(model, likelihood_loss_function_coefficients)
        solution = NonLinearSolver.default().solve(problem, self.profiler())
        model.update_gammas_from(solution)

    def estimate(self, model, transactions):
        likelihood_loss_function_coefficients = self.likelihood_loss_function_coefficients(transactions)
        new_likelihood = model.log_likelihood_for(transactions)

        max_iterations = len(likelihood_loss_function_coefficients)
        cpu_time = time_for_optimization(partial_time=Settings.instance().non_linear_solver_partial_time_limit(),
                                         total_time=Settings.instance().solver_total_time_limit(),
                                         profiler=self.profiler())
        start_time = time.time()

        for _ in range(max_iterations):
            old_likelihood = new_likelihood

            possible_mnl_model = self.look_for_new_mnl_model(model, likelihood_loss_function_coefficients)
            model.add_new_class_with(possible_mnl_model)
            self.update_weights_for(model, likelihood_loss_function_coefficients)

            new_likelihood = model.log_likelihood_for(transactions)

            likelihood_does_not_increase = new_likelihood < old_likelihood
            likelihood_does_not_increase_enough = abs(new_likelihood - old_likelihood) / len(transactions) < 1e-7
            time_limit = (time.time() - start_time) > cpu_time

            if likelihood_does_not_increase or likelihood_does_not_increase_enough or time_limit:
                break

        return model


class NewMNLSubProblem(NonLinearProblem):
    def __init__(self, latent_class_model, possible_mnl_model, likelihood_loss_function_coefficients):
        self.latent_class_model = latent_class_model
        self.likelihood_loss_function_coefficients = likelihood_loss_function_coefficients
        self.likelihood_loss_function_gradient = self.compute_likelihood_loss_function_gradient()
        self.possible_mnl_model = possible_mnl_model

    def compute_likelihood_loss_function_gradient(self):
        gradient = []
        for transaction, number_sales in self.likelihood_loss_function_coefficients:
            probability = self.latent_class_model.probability_of(transaction)
            gradient.append((transaction, - (number_sales / probability)))
        return gradient

    def constraints(self):
        return self.possible_mnl_model.constraints()

    def objective_function(self, parameters):
        self.possible_mnl_model.update_parameters_from_vector(parameters)
        result = 0
        for transaction, gradient_component in self.likelihood_loss_function_gradient:
            result += (gradient_component * self.possible_mnl_model.probability_of(transaction))
        return result / len(self.likelihood_loss_function_coefficients)

    def amount_of_variables(self):
        return len(self.possible_mnl_model.parameters_vector())

    def initial_solution(self):
        return array(self.possible_mnl_model.parameters_vector())


class NewWeightsSubProblem(NonLinearProblem):
    def __init__(self, model, likelihood_loss_function_coefficients):
        self.model = model
        self.likelihood_loss_function_coefficients = likelihood_loss_function_coefficients

    def constraints(self):
        return NewWeightsConstraints(self.model)

    def objective_function(self, vector):
        self.model.update_gammas_from(vector)
        result = 0.0
        for transaction, number_sales in self.likelihood_loss_function_coefficients:
            result -= (number_sales * safe_log(self.model.probability_of(transaction)))
        return result / len(self.likelihood_loss_function_coefficients)

    def amount_of_variables(self):
        return self.model.amount_of_classes()

    def initial_solution(self):
        return array(self.model.gammas)


class NewWeightsConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(self.model.amount_of_classes()) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(self.model.amount_of_classes()) * ONE_UPPER_BOUND

    def amount_of_constraints(self):
        return 1

    def lower_bounds_over_constraints_vector(self):
        return array([1.0])

    def upper_bounds_over_constraints_vector(self):
        return array([1.0])

    def non_zero_parameters_on_constraints_jacobian(self):
        return self.model.amount_of_classes()

    def constraints_evaluator(self):
        def evaluator(x):
            return array([sum(x)])
        return evaluator

    def constraints_jacobian_evaluator(self):
        def jacobian_evaluator(x, flag):
            if flag:
                return (zeros(len(self.model.gammas)),
                        array(list(range(len(self.model.gammas)))))
            else:
                return ones(len(self.model.gammas))

        return jacobian_evaluator
