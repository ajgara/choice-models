# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from numpy import array
from estimation.expectation_maximization import ExpectationMaximizationEstimator
from models.latent_class import LatentClassModel
from optimization.non_linear import NonLinearProblem, NonLinearSolver
from profiler import Profiler
import copy


class LatentClassExpectationMaximizationEstimator(ExpectationMaximizationEstimator):
    """
        Taken from "Discrete Choice Methods with Simulation" by Kenneth E. Train (Second Edition Chapter 14).
    """
    def one_step(self, model, transactions):
        total_weights = 0.0
        weights = []

        lc_cache = {}
        for klass_share, klass_model in zip(model.gammas, model.mnl_models()):
            klass_transactions_weights = []
            mnl_cache = {}
            for transaction in transactions:
                memory = (transaction.product, tuple(transaction.offered_products))
                if memory in lc_cache:
                    lc_probability = lc_cache[memory]
                else:
                    lc_probability = model.probability_of(transaction)
                    lc_cache[memory] = lc_probability

                if memory in mnl_cache:
                    mnl_probability = mnl_cache[memory]
                else:
                    mnl_probability = klass_model.probability_of(transaction)
                    mnl_cache[memory] = mnl_probability

                numerator = (klass_share * mnl_probability)
                denominator = lc_probability
                probability = numerator / denominator
                total_weights += probability
                klass_transactions_weights.append(probability)
            weights.append(klass_transactions_weights)

        new_gammas = []
        for klass_transactions_weights in weights:
            new_gammas.append(sum(klass_transactions_weights) / total_weights)

        new_models = []
        for klass_transactions_weights, klass_model in zip(weights, model.mnl_models()):
            initial = copy.deepcopy(klass_model)
            problem = WeightedMultinomialLogitMaximumLikelihoodNonLinearProblem(initial, transactions,
                                                                                klass_transactions_weights)
            solution = NonLinearSolver.default().solve(problem, Profiler(verbose=False))
            initial.update_parameters_from_vector(solution)
            new_models.append(initial)

        return LatentClassModel(products=model.products, gammas=new_gammas, multi_logit_models=new_models)


class WeightedMultinomialLogitMaximumLikelihoodNonLinearProblem(NonLinearProblem):
    def __init__(self, model, transactions, transactions_weights):
        self.model = model
        self.transactions = transactions
        self.transaction_weights = transactions_weights

    def constraints(self):
        return self.model.constraints()

    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        result = 0.0
        cache = {}
        for weight, transaction in zip(self.transaction_weights, self.transactions):
            memory = (transaction.product, tuple(transaction.offered_products))
            if memory in cache:
                log_probability = cache[memory]
            else:
                log_probability = self.model.log_probability_of(transaction)
                cache[memory] = log_probability
            result += (weight * log_probability)
        return -result / len(self.transactions)

    def initial_solution(self):
        return array(self.model.parameters_vector())

    def amount_of_variables(self):
        return len(self.model.parameters_vector())
