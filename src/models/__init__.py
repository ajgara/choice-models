# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from math import log, sqrt
from transactions.base import Transaction
from utils import safe_log
import json


class Model(object):
    """
        Represents a mathematical model for Discrete Choice Consumer Decision.
    """
    def __init__(self, products):
        if products != list(range(len(products))):
            raise Exception('Products should be entered as an ordered consecutive list.')
        self.products = products

    @classmethod
    def code(cls):
        raise NotImplementedError('Subclass responsibility')

    @classmethod
    def from_data(cls, data):
        for klass in cls.__subclasses__():
            if data['code'] == klass.code():
                return klass.from_data(data)
        raise Exception('No model can be created from data %s')

    @classmethod
    def simple_deterministic(cls, *args, **kwargs):
        """
            Must return a default model with simple pdf parameters to use as an initial solution for estimators.
        """
        raise NotImplementedError('Subclass responsibility')

    @classmethod
    def simple_random(cls, *args, **kwargs):
        """
            Must return a default model with random pdf parameters to use as a ground model.
        """
        raise NotImplementedError('Subclass responsibility')

    def probability_of(self, transaction):
        """
            Must return the probability of a transaction.
        """
        raise NotImplementedError('Subclass responsibility')

    def log_probability_of(self, transaction):
        return safe_log(self.probability_of(transaction))

    def probability_distribution_over(self, offered_products):
        distribution = []
        for product in range(len(self.products)):
            transaction = Transaction(product, offered_products)
            distribution.append(self.probability_of(transaction))
        return distribution

    def log_likelihood_for(self, transactions):
        result = 0
        cache = {}
        for transaction in transactions:
            cache_code = (transaction.product, tuple(transaction.offered_products))
            if cache_code in cache:
                log_probability = cache[cache_code]
            else:
                log_probability = self.log_probability_of(transaction)
                cache[cache_code] = log_probability
            result += log_probability
        return result / len(transactions)

    def infinite_in_sample_log_likelihood(self, ground_model):
        result = 0
        for t in Transaction.all_for(self):
            result += (ground_model.probability_of(t) * self.log_probability_of(t))
        return result

    def soft_rmse_for(self, ground_model):
        rmse = 0.0
        amount_terms = 0.0
        for t in Transaction.all_for(self):
            rmse += ((self.probability_of(t) - ground_model.probability_of(t)) ** 2)
            amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def rmse_for(self, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability = self.probability_of(Transaction(product, transaction.offered_products))
                rmse += ((probability - float(product == transaction.product)) ** 2)
                amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def rmse_known_ground(self, ground_model, transactions):
        rmse = 0.0
        amount_terms = 0
        for transaction in transactions:
            for product in transaction.offered_products:
                probability_1 = self.probability_of(Transaction(product, transaction.offered_products))
                probability_2 = ground_model.probability_of(Transaction(product, transaction.offered_products))
                rmse += ((probability_1 - probability_2) ** 2)
                amount_terms += 1
        return sqrt(rmse / float(amount_terms))

    def hit_rate_for(self, transactions):
        hit_rate = 0
        for transaction in transactions:
            probabilities = []
            for product in transaction.offered_products:
                probabilities.append((product, self.probability_of(Transaction(product, transaction.offered_products))))
            most_probable = max(probabilities, key=lambda p: p[1])[0]
            hit_rate += int(most_probable == transaction.product)
        return float(hit_rate) / float(len(transactions))

    def soft_chi_squared_score_for(self, ground_model, transactions):
        expected_purchases = [0.0 for _ in self.products]
        observed_purchases = [0.0 for _ in self.products]

        for transaction in transactions:
            for product in transaction.offered_products:
                observed_purchases[product] += ground_model.probability_of(Transaction(product, transaction.offered_products))
                expected_purchases[product] += self.probability_of(Transaction(product, transaction.offered_products))

        score = 0.0
        for p in self.products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(self.products))

    def hard_chi_squared_score_for(self, transactions):
        expected_purchases = [0.0 for _ in self.products]
        observed_purchases = [0.0 for _ in self.products]

        for transaction in transactions:
            observed_purchases[transaction.product] += 1.0
            for product in transaction.offered_products:
                expected_purchases[product] += self.probability_of(Transaction(product, transaction.offered_products))

        score = 0.0
        for p in self.products:
            score += (((expected_purchases[p] - observed_purchases[p]) ** 2) / (expected_purchases[p] + 0.5))
        return score / float(len(self.products))

    def aic_for(self, transactions):
        k = self.amount_of_parameters()
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions) * len(transactions)
        return 2 * (k - l + (k * (k + 1) / (amount_samples - k - 1)))

    def bic_for(self, transactions):
        k = self.amount_of_parameters()
        amount_samples = len(transactions)
        l = self.log_likelihood_for(transactions) * len(transactions)
        return -2 * l + (k * log(amount_samples))

    def amount_of_parameters(self):
        return len(self.parameters_vector())

    def save(self, file_name):
        with open(file_name, 'w+') as f:
            f.write(json.dumps(self.data(), indent=1))

    @classmethod
    def load(cls, file_name):
        with open(file_name, 'r') as f:
            model = cls.from_data(json.loads(f.read()))
        return model

    def parameters_vector(self):
        """
            Vector of parameters that define the model. For example lambdas and ros in Markov Chain.
        """
        return []

    def update_parameters_from_vector(self, parameters):
        """
            Updates current parameters from input parameters vector
        """
        pass

    def constraints(self):
        raise NotImplementedError('Subclass responsibility')

    def data(self):
        raise NotImplementedError('Subclass responsibility')


from models.exponomial import ExponomialModel
from models.latent_class import LatentClassModel
from models.markov_chain import MarkovChainModel
from models.markov_chain_rank_2 import MarkovChainRank2Model
from models.multinomial_logit import MultinomialLogitModel
from models.nested_logit import NestedLogitModel
from models.ranked_list import RankedListModel
from models.random_choice import RandomChoiceModel
from models.mixed_logit import MixedLogitModel
from models.generalized_stochastic_preference import GeneralizedStochasticPreferenceModel
