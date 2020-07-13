
from numpy import linalg, ones, array
from models import Model
from utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_one, ZERO_LOWER_BOUND, \
    ONE_UPPER_BOUND, rindex
from optimization.non_linear import Constraints


class MarkovChainModel(Model):
    @classmethod
    def code(cls):
        return 'mkv'

    @classmethod
    def from_data(cls, data, l_mask=None, p_mask=None):
        lambdas_mask = data['lambdas_mask']
        lambdas = data['lambdas']
        ros_mask = data['ros_mask']
        ros = data['ros']
        return cls(data['products'], lambdas, ros, l_mask=lambdas_mask, p_mask=ros_mask)

    @classmethod
    def simple_deterministic(cls, products):
        lambdas = generate_n_equal_numbers_that_sum_one(len(products))
        ros = []
        for i in range(len(products)):
            row = generate_n_equal_numbers_that_sum_one(len(products) - 1)
            ros.append(row[:i] + [0] + row[i:])

        return cls(products, lambdas, ros)

    @classmethod
    def simple_random(cls, products):
        lambdas = generate_n_random_numbers_that_sum_one(len(products))

        ros = []
        for i in range(len(products)):
            row = generate_n_random_numbers_that_sum_one(len(products) - 1)
            ros.append(row[:i] + [0] + row[i:])

        return cls(products, lambdas, ros)

    @classmethod
    def from_mnl(cls, mnl_model):
        summ = sum(mnl_model.etas) + 1.0
        new_etas = [1.0 / summ] + [eta / summ for eta in mnl_model.etas]

        lambdas = [eta for eta in new_etas]
        ros = [[eta_j / (1.0 - eta_i) for eta_j in new_etas] for eta_i in new_etas]
        return cls(mnl_model.products, lambdas, ros)

    def __init__(self, products, l, p, l_mask=None, p_mask=None):
        super(MarkovChainModel, self).__init__(products)
        self._l_mask = l_mask if l_mask else [1 for _ in products]
        self._p_mask = p_mask if p_mask else [[1 for _ in products] for _ in products]

        if len(l) != len(products):
            info = (len(l), len(products))
            raise Exception('Amount of lambdas (%s) should be equal to amount of products (%s).' % info)
        if any([len(l) != len(a) for a in p]) or len(p) != len(products):
            raise Exception('Ro matrix should be squared and of size amount of products (%s).' % len(products))
        if len(self._l_mask) != len(products):
            raise Exception('Invalid lambda mask')
        if any([len(products) != len(a) for a in self._p_mask]) or len(self._p_mask) != len(products):
            raise Exception('Invalid ros mask')

        self._l = l
        self._p = p
        self._theta_cafe = {}

    def clean_cache(self):
        self._theta_cafe = {}

    def is_in_cache(self, key):
        return key in self._theta_cafe

    def cache_value_for(self, key):
        return self._theta_cafe[key]

    def save_in_cache(self, key, value):
        self._theta_cafe[key] = value

    def lambdas(self):
        return self._l

    def ros(self):
        return self._p

    def lambda_for(self, product):
        return self._l[product]

    def ro_for(self, product_i, product_j):
        return self._p[product_i][product_j]

    def set_lambdas(self, new_lambdas):
        self.clean_cache()
        self._l = new_lambdas

    def set_ros(self, new_ros):
        self.clean_cache()
        self._p = new_ros

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0
        return self.expected_number_of_visits_if(transaction.offered_products)[transaction.product]

    def expected_number_of_visits_if(self, offered_products):
        if self.is_in_cache(tuple(offered_products)):
            return self.cache_value_for(tuple(offered_products))

        not_offered_products = sorted(list(set(self.products) - set(offered_products)))

        theta = [self._l[product] for product in self.products]

        if len(not_offered_products):
            A = []
            b = []
            for product_i in not_offered_products:
                row = []
                for product_j in not_offered_products:
                    if product_i == product_j:
                        row.append(1.0)
                    else:
                        row.append(-self._p[product_j][product_i])
                A.append(row)
                b.append(self._l[product_i])

            solution = linalg.solve(A, b)

            for solution_index, product in enumerate(not_offered_products):
                theta[product] = solution[solution_index]

            for product in offered_products:
                theta[product] = self._l[product] + sum([self._p[n_product][product] * theta[n_product] for n_product in not_offered_products])

        self.save_in_cache(tuple(offered_products), theta)
        return theta

    def parameters_vector(self):
        pars = [l for l, m in zip(self._l, self._l_mask) if m][:-1]
        for r, rm in zip(self._p, self._p_mask):
            pars += [p for p, m in zip(r, rm) if m][:-1]
        return pars

    def update_parameters_from_vector(self, parameters):
        self.clean_cache()

        paramenters_index = 0
        accum = 0
        sum_index = rindex(self._l_mask, 1)

        for i, m in enumerate(self._l_mask):
            if sum_index == i:
                self._l[i] = 1.0 - accum
            else:
                self._l[i] = parameters[paramenters_index] if m else 0
                accum += self._l[i]
                paramenters_index += 1 if m else 0

        for i, row_mask in enumerate(self._p_mask):
            accum = 0
            sum_index = rindex(row_mask, 1)

            for j, m in enumerate(row_mask):
                if sum_index == j:
                    self._p[i][j] = 1.0 - accum
                else:
                    self._p[i][j] = parameters[paramenters_index] if m else 0
                    accum += self._p[i][j]
                    paramenters_index += 1 if m else 0

        return parameters

    def constraints(self):
        return MarkovChainModelConstraints(self)

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'lambdas_mask': self._l_mask,
            'lambdas': self._l,
            'ros_mask': self._p_mask,
            'ros': self._p
        }

    def __repr__(self):
        ros = '\n'.join([','.join(map(str, row)) for row in self.ros()])
        return '<Products: %s ;\n l: %s ;\n p:\n %s>' % (self.products, self.lambdas(), ros)


class MarkovChainModelConstraints(Constraints):
    def __init__(self, model):
        self.model = model

    def lower_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ZERO_LOWER_BOUND

    def upper_bounds_vector(self):
        return ones(len(self.model.parameters_vector())) * ONE_UPPER_BOUND

    def amount_of_constraints(self):
        constraints = 1 if sum(self.model._l_mask) > 1 else 0
        constraints += sum([1 for row in self.model._p_mask if sum(row) > 1])
        return constraints

    def lower_bounds_over_constraints_vector(self):
        return ones(self.amount_of_constraints()) * ZERO_LOWER_BOUND

    def upper_bounds_over_constraints_vector(self):
        return ones(self.amount_of_constraints()) * ONE_UPPER_BOUND

    def non_zero_parameters_on_constraints_jacobian(self):
        return len(self.model.parameters_vector())

    def constraints_evaluator(self):
        def evaluator(x):
            constraints = []
            i = 0
            constraint = 0.0
            for _ in range(sum(self.model._l_mask) - 1):
                constraint += x[i]
                i += 1

            if sum(self.model._l_mask) > 1:
                constraints.append(constraint)

            for row in self.model._p_mask:
                constraint = 0.0
                for _ in range(sum(row) - 1):
                    constraint += x[i]
                    i += 1
                if sum(row) > 1:
                    constraints.append(constraint)
            return array(constraints)
        return evaluator

    def constraints_jacobian_evaluator(self):
        def jacobian_evaluator(x, flag):
            if flag:
                sparse_rows = [0] * (sum(self.model._l_mask) - 1) if sum(self.model._l_mask) > 1 else []
                j = 1
                for row in self.model._p_mask:
                    if sum(row) > 1:
                        sparse_rows = sparse_rows + [j] * (sum(row) - 1)
                        j += 1
                sparse_columns = [i for i in range(len(sparse_rows))]
                return array(sparse_rows), array(sparse_columns)
            else:
                return ones(len(self.model.parameters_vector()))
        return jacobian_evaluator
