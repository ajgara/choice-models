# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.


from numpy import ones, array, zeros
from python_choice_models.models import Model
from python_choice_models.utils import generate_n_random_numbers_that_sum_one, generate_n_equal_numbers_that_sum_one, ZERO_LOWER_BOUND, \
    ONE_UPPER_BOUND
from python_choice_models.optimization.non_linear import Constraints
import random


class GeneralizedStochasticPreferenceModel(Model):
    @classmethod
    def code(cls):
        return 'gsp'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['ranked_lists'], data['betas'], data['choices'])

    @classmethod
    def simple_deterministic(cls, products, ranked_lists):
        betas = generate_n_equal_numbers_that_sum_one(len(ranked_lists))[1:]
        choices = [1 for _ in range(len(ranked_lists))]
        return cls(products, ranked_lists, betas, choices)

    @classmethod
    def simple_random(cls, products, ranked_lists):
        betas = generate_n_random_numbers_that_sum_one(len(ranked_lists))[1:]
        choices = [random.choice(products) for _ in range(len(ranked_lists))]
        return cls(products, ranked_lists, betas, choices)

    def __init__(self, products, ranked_lists, betas, choices):
        super(GeneralizedStochasticPreferenceModel, self).__init__(products)
        if len(betas) + 1 != len(ranked_lists):
            info = (len(betas), len(ranked_lists))
            raise Exception('Amount of betas (%s) should be one less than of ranked lists (%s).' % info)
        if any([len(ranked_list) != len(products) for ranked_list in ranked_lists]):
            info = (products, ranked_lists)
            raise Exception('All ranked list should have all products.\n Products: %s\n Ranked lists: %s\n' % info)
        if len(choices) != len(ranked_lists):
            raise Exception('Amount of ranked lists and choices must be the same')

        self.choices = choices
        self.ranked_lists = ranked_lists
        self.betas = betas

    def chosen_product_for(self, choice, ranked_list, offered_product):
        only_offered = [p for p in ranked_list if p in offered_product]
        trimmed = only_offered[:only_offered.index(0)] + [0]
        if choice - 1 >= len(trimmed) or choice == 0:
            return 0
        return trimmed[choice - 1]

    def probability_of(self, transaction):
        probability = 0
        for ranked_list_number, choice, ranked_list in zip(list(range(len(self.ranked_lists))),
                                                           self.choices,
                                                           self.ranked_lists):
            chosen_product = self.chosen_product_for(choice, ranked_list, transaction.offered_products)
            if chosen_product == transaction.product:
                probability += self.beta_for(ranked_list_number)
        return probability

    def amount_of_ranked_lists(self):
        return len(self.ranked_lists)

    def beta_for(self, ranked_list_number):
        return 1 - sum(self.betas) if ranked_list_number == 0 else self.betas[ranked_list_number - 1]

    def parameters_vector(self):
        return self.betas

    def update_parameters_from_vector(self, parameters):
        self.betas = list(parameters)

    def constraints(self):
        return GeneralizedStochasticPreferenceModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'betas': self.betas,
            'ranked_lists': self.ranked_lists,
            'choices': self.choices,
        }

    def __repr__(self):
        return '<Products: %s ; Ranked Lists: %s ; Betas: %s ; Choices: %s >' % (self.products, self.ranked_lists, self.betas, self.choices)


class GeneralizedStochasticPreferenceModelConstraints(Constraints):
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
