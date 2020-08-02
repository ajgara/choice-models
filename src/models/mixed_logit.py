# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

import ghalton
from models import Model
from optimization.non_linear import Constraints
from utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_one, ZERO_LOWER_BOUND
from scipy.stats import norm
import numpy as np
import scipy


class MixedLogitModel(Model):
    @classmethod
    def code(cls):
        return 'mx'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['mus'], data['sigmas'])

    @classmethod
    def simple_random(cls, products):
        mus = generate_n_random_numbers_that_sum_one(len(products))
        sigmas = generate_n_random_numbers_that_sum_one(len(products))
        return cls(products, mus, sigmas)

    @classmethod
    def simple_deterministic(cls, products):
        mus = generate_n_equal_numbers_that_sum_one(len(products))
        sigmas = generate_n_equal_numbers_that_sum_one(len(products))
        return cls(products, mus, sigmas)

    def __init__(self, products, mus, sigmas):
        super(MixedLogitModel, self).__init__(products)
        if len(mus) != len(products):
            raise Exception('Mus should be equal to amount of products.')
        if len(sigmas) != len(products):
            raise Exception('Sigmas should be equal amount of products.')
        self.products = products
        self.mus = mus
        self.sigmas = sigmas

        self.NUMBER_SAMPLES = 1000
        self.random_numbers = ghalton.Halton(len(self.products)).get(self.NUMBER_SAMPLES)
        self.random_numbers = norm.ppf(self.random_numbers, loc=0.0, scale=1.0)

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0

        utilities = np.dot(self.random_numbers, np.diagflat(np.sqrt(self.sigmas)))
        utilities += np.dot(np.ones((self.NUMBER_SAMPLES, 1)), np.array([self.mus]))
        utilities = np.exp(utilities)

        offer_set_matrix = np.array([[1.0 if p in transaction.offered_products else 0.0 for p in self.products]]).T
        purchase_matrix = np.array([[1.0 if p == transaction.product else 0.0 for p in self.products]]).T

        denominators = np.dot(utilities, offer_set_matrix)
        numerators = np.dot(utilities, purchase_matrix)
        transaction_probs = np.dot(np.ones((1, self.NUMBER_SAMPLES)), (numerators / denominators)) / self.NUMBER_SAMPLES

        return transaction_probs[0][0]

    def probabilities_of(self, transactions):
        utilities = np.dot(self.random_numbers, np.diagflat(np.sqrt(self.sigmas)))
        utilities += np.dot(np.ones((self.NUMBER_SAMPLES, 1)), np.array([self.mus]))
        utilities = np.exp(utilities)

        offer_set_matrix = np.array(
            [[1.0 if p in t.offered_products else 0.0 for p in self.products] for t in transactions]).T
        purchase_matrix = np.array([[1.0 if p == t.product else 0.0 for p in self.products] for t in transactions]).T

        denominators = np.dot(utilities, offer_set_matrix)
        numerators = np.dot(utilities, purchase_matrix)

        transaction_probs = np.dot(np.ones((1, self.NUMBER_SAMPLES)), (numerators / denominators)) / self.NUMBER_SAMPLES
        return transaction_probs

    def log_likelihood_for(self, transactions):
        transaction_probs = self.probabilities_of(transactions)
        return np.sum(np.log(transaction_probs)) / len(transactions)

    def parameters_vector(self):
        return self.mus + self.sigmas

    def update_parameters_from_vector(self, parameters):
        self.mus = list(parameters)[:len(self.products)]
        self.sigmas = list(parameters)[len(self.products):]

    def constraints(self):
        return MixedLogitModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'mus': self.mus,
            'sigmas': self.sigmas,
        }


class MixedLogitModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        lower_bounds = [None for _ in self.model.products]
        lower_bounds += [ZERO_LOWER_BOUND for _ in self.model.products]
        return np.ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return [None for _ in range(len(self.model.products) * 2)]
