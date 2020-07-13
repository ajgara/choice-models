# coding=utf-8
from numpy import ones
from models import Model
from optimization.non_linear import Constraints
import numpy as np
import settings


class ExponomialModel(Model):
    @classmethod
    def code(cls):
        return 'exp'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['utilities'])

    @classmethod
    def simple_deterministic(cls, products):
        return cls(products, list(np.random.uniform(-1.0, 1.0, len(products))))

    @classmethod
    def simple_random(cls, products):
        return cls(products, list(np.random.uniform(-1.0, 1.0, len(products))))

    def __init__(self, products, utilities):
        super(ExponomialModel, self).__init__(products)
        if len(products) != len(utilities):
            info = (len(products), len(utilities))
            raise Exception('Given number of utilities (%s) does not match number of products (%s).' % info)
        self.utilities = utilities

    def utility_for(self, product):
        return self.utilities[product]

    def g(self, product, offered_products):
        better_products = [p for p in offered_products if self.utility_for(p) >= self.utility_for(product)]
        num = np.exp(-sum([self.utility_for(p) - self.utility_for(product) for p in better_products]))
        return num / len(better_products)

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0

        worse_products = [p for p in transaction.offered_products if self.utility_for(p) < self.utility_for(transaction.product)]
        worse_products = sorted(worse_products, key=lambda p: self.utility_for(p))

        accum = self.g(transaction.product, transaction.offered_products)
        for k, product in enumerate(worse_products):
            accum -= (1.0 / (len(transaction.offered_products) - k - 1.0)) * self.g(product,
                                                                                    transaction.offered_products)

        return accum

    def parameters_vector(self):
        return self.utilities

    def update_parameters_from_vector(self, parameters):
        self.utilities = list(parameters)

    def constraints(self):
        return ExponomialModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'utilities': self.utilities,
        }

    def __repr__(self):
        return '<Products: %s ; Utilities: %s >' % (self.products, self.utilities)


class ExponomialModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * settings.NLP_LOWER_BOUND_INF

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * settings.NLP_UPPER_BOUND_INF
