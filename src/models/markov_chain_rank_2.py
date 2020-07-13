from models import Model
from optimization.non_linear import Constraints
from settings import NLP_UPPER_BOUND_INF
from utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_one, ZERO_LOWER_BOUND, \
    ONE_UPPER_BOUND, generate_n_equal_numbers_that_sum_m
import numpy as np


class MarkovChainRank2Model(Model):
    @classmethod
    def code(cls):
        return 'mkv2'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['l'], data['u'], data['v'], data['w'], data['z'])

    @classmethod
    def simple_deterministic(cls, products):
        l = generate_n_equal_numbers_that_sum_one(len(products))
        u = [1.0 for _ in products]
        v = generate_n_equal_numbers_that_sum_m(len(products), m=1.0)
        w = [1.0 for _ in products]
        z = generate_n_equal_numbers_that_sum_m(len(products), m=0.0)
        return cls(products, l, u, v, w, z)

    @classmethod
    def simple_random(cls, products):
        l = generate_n_random_numbers_that_sum_one(len(products))
        u = [1.0 for _ in products]
        v = generate_n_equal_numbers_that_sum_m(len(products), m=0.5)
        w = [1.0 for _ in products]
        z = generate_n_equal_numbers_that_sum_m(len(products), m=0.5)
        return cls(products, l, u, v, w, z)

    def __init__(self, products, l, u, v, w, z):
        super(MarkovChainRank2Model, self).__init__(products)
        self._l = l
        self._u = u
        self._v = v
        self._w = w
        self._z = z
        self._theta_cafe = {}

        if any([len(x) != len(products) for x in [l, u, v, w, z]]):
            info = ([len(x) for x in [l, u, v, w, z]], len(products))
            raise Exception('Size of vectors (%s) should be equal to amount of products (%s).' % info)

    def clean_cache(self):
        self._theta_cafe = {}

    def is_in_cache(self, key):
        return key in self._theta_cafe

    def cache_value_for(self, key):
        return self._theta_cafe[key]

    def save_in_cache(self, key, value):
        self._theta_cafe[key] = value

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0

        expected_vists = self.expected_number_of_visits_if(transaction.offered_products)
        if min(transaction.offered_products) == transaction.product:
            return 1.0 - sum([expected_vists[p] for p in sorted(transaction.offered_products)[1:]])
        return expected_vists[transaction.product]

    def lambdas(self):
        return self._l

    def ros(self):
        s = len(self.products)

        u = np.array(self._u).reshape((s, 1))
        v = np.array(self._v).reshape((s, 1))
        w = np.array(self._w).reshape((s, 1))
        z = np.array(self._z).reshape((s, 1))

        return np.dot(u, v.T) + np.dot(w, z.T)

    def expected_number_of_visits_if(self, offered_products):
        if self.is_in_cache(tuple(offered_products)):
            return self.cache_value_for(tuple(offered_products))

        l = self.lambdas()
        ro = self.ros()

        not_offered_products = sorted(list(set(self.products) - set(offered_products)))
        theta = [l[product] for product in self.products]

        if len(not_offered_products):
            A = []
            b = []
            for product_i in not_offered_products:
                row = []
                for product_j in not_offered_products:
                    if product_i == product_j:
                        row.append(1.0)
                    else:
                        row.append(-ro[product_j][product_i])
                A.append(row)
                b.append(l[product_i])

            solution = np.linalg.solve(A, b)

            for solution_index, product in enumerate(not_offered_products):
                theta[product] = solution[solution_index]

            for product in offered_products:
                theta[product] = l[product] + sum([ro[n_product][product] * theta[n_product] for n_product in not_offered_products])

        self.save_in_cache(tuple(offered_products), theta)
        return theta

    def parameters_vector(self):
        return list(self._l) + list(self._u) + list(self._v) + list(self._w) + list(self._z)

    def update_parameters_from_vector(self, parameters):
        self.clean_cache()

        i = len(self.products)
        self._l = parameters[i*0:i*1]
        self._u = parameters[i*1:i*2]
        self._v = parameters[i*2:i*3]
        self._w = parameters[i*3:i*4]
        self._z = parameters[i*4:i*5]

        return parameters

    def constraints(self):
        return MarkovChainRank2ModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'l': list(self._l),
            'u': list(self._u),
            'v': list(self._v),
            'w': list(self._w),
            'z': list(self._z)
        }

    def __repr__(self):
        info = (self.products, self._l, self._u, self._v, self._w, self._z)
        return '<Products: %s ;\n l: %s ;\n u: %s ;\n v: %s ;\n w: %s ;\n z: %s ; >' % info


class MarkovChainRank2ModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return np.ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return np.ones(len(self.model.parameters_vector())) * NLP_UPPER_BOUND_INF

    def amount_of_constraints(self):
        return len(self.model.products) + 1

    def lower_bounds_over_constraints_vector(self):
        return np.ones(self.amount_of_constraints()) * ONE_UPPER_BOUND

    def upper_bounds_over_constraints_vector(self):
        return np.ones(self.amount_of_constraints()) * ONE_UPPER_BOUND

    def constraints_evaluator(self):
        def evaluator(x):
            i = len(self.model.products)

            l = np.array(x[i*0:i*1]).reshape((i, 1))
            u = np.array(x[i*1:i*2]).reshape((i, 1))
            v = np.array(x[i*2:i*3]).reshape((i, 1))
            w = np.array(x[i*3:i*4]).reshape((i, 1))
            z = np.array(x[i*4:i*5]).reshape((i, 1))

            ro = np.dot(u, v.T) + np.dot(w, z.T)

            return list(np.dot(ro, np.ones((i, 1)))) + [np.sum(l)]
        return evaluator
