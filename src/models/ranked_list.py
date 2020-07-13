from numpy import ones, array, zeros
from models import Model
from utils import generate_n_random_numbers_that_sum_one, generate_n_equal_numbers_that_sum_one, ZERO_LOWER_BOUND, \
    ONE_UPPER_BOUND
from optimization.non_linear import Constraints
import numpy as np


class RankedListModel(Model):
    @classmethod
    def code(cls):
        return 'rl'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['ranked_lists'], data['betas'])

    @classmethod
    def simple_deterministic(cls, products, ranked_lists):
        betas = generate_n_equal_numbers_that_sum_one(len(ranked_lists))[1:]
        return cls(products, ranked_lists, betas)

    @classmethod
    def simple_random(cls, products, ranked_lists):
        betas = generate_n_random_numbers_that_sum_one(len(ranked_lists))[1:]
        return cls(products, ranked_lists, betas)

    @classmethod
    def simple_deterministic_independent(cls, products):
        ranked_lists = [[i] + sorted(set(products) - {i}) for i in range(len(products))]
        return cls.simple_deterministic(products, ranked_lists)

    def __init__(self, products, ranked_lists, betas):
        super(RankedListModel, self).__init__(products)
        if len(betas) + 1 != len(ranked_lists):
            info = (len(betas), len(ranked_lists))
            raise Exception('Amount of betas (%s) should be one less than of ranked lists (%s).' % info)
        if any([len(ranked_list) != len(products) for ranked_list in ranked_lists]):
            info = (products, ranked_lists)
            raise Exception('All ranked list should have all products.\n Products: %s\n Ranked lists: %s\n' % info)

        self.ranked_lists = ranked_lists
        self.betas = betas

    def compatibility_matrix_for(self, transactions):
        matrix = []
        for t in transactions:
            matrix.append([1.0 if self.are_compatible(r, t) else 0.0 for r in self.ranked_lists])
        return np.array(matrix)

    def probabilities_for(self, transactions):
        return np.dot(self.compatibility_matrix_for(transactions), np.array(self.all_betas()))

    def probability_of(self, transaction):
        probability = 0
        for ranked_list_number, ranked_list in self.ranked_lists_compatible_with(transaction):
            probability += self.beta_for(ranked_list_number)
        return probability

    def amount_of_ranked_lists(self):
        return len(self.ranked_lists)

    def all_betas(self):
        return [self.beta_for(ranked_list_number) for ranked_list_number in range(len(self.ranked_lists))]

    def beta_for(self, ranked_list_number):
        return 1 - sum(self.betas) if ranked_list_number == 0 else self.betas[ranked_list_number - 1]

    def set_betas(self, all_betas):
        self.betas = all_betas[1:]

    def are_compatible(self, ranked_list, transaction):
        if transaction.product not in ranked_list:
            return False
        better_products = ranked_list[:ranked_list.index(transaction.product)]
        return all([p not in transaction.offered_products for p in better_products])

    def ranked_lists_compatible_with(self, transaction):
        if transaction.product not in transaction.offered_products:
            return []

        compatible_ranked_lists = []
        for i, ranked_list in enumerate(self.ranked_lists):
            if self.are_compatible(ranked_list, transaction):
                compatible_ranked_lists.append((i, ranked_list))
        return compatible_ranked_lists

    def add_ranked_list(self, ranked_list):
        if ranked_list not in self.ranked_lists:
            percentage = 1.0 / (len(self.betas) + 2.0)
            new_beta = sum([beta * percentage for beta in self.all_betas()])
            self.betas = [beta * (1.0 - percentage) for beta in self.betas] + [new_beta]
            self.ranked_lists.append(ranked_list)

    def parameters_vector(self):
        return self.betas

    def update_parameters_from_vector(self, parameters):
        self.betas = list(parameters)

    def constraints(self):
        return RankedListModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'betas': self.betas,
            'ranked_lists': self.ranked_lists,
        }

    def __repr__(self):
        return '<Products: %s ; Ranked Lists: %s ; Betas: %s >' % (self.products, self.ranked_lists, self.betas)


class RankedListModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector()))

    def amount_of_constraints(self):
        return 1

    def lower_bounds_over_constraints_vector(self):
        return array([ZERO_LOWER_BOUND])

    def upper_bounds_over_constraints_vector(self):
        return array([ONE_UPPER_BOUND])

    def non_zero_parameters_on_constraints_jacobian(self):
        return len(self.model.betas)

    def constraints_evaluator(self):
        def evaluator(x):
            return array([sum(x)])
        return evaluator

    def constraints_jacobian_evaluator(self):
        def jacobian_evaluator(x, flag):
            if flag:
                return zeros(len(self.model.betas)), array(list(range(len(self.model.betas))))
            else:
                return ones(len(self.model.betas))
        return jacobian_evaluator
