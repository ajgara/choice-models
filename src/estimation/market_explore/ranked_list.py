# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from estimation.market_explore import MarketExplorer
from optimization.linear import LinearProblem, LinearSolver
from copy import deepcopy


class RankedListMarketExplorer(MarketExplorer):
    @classmethod
    def code(cls):
        raise NotImplementedError('Subclass responsibility')

    def explore_for(self, estimator, model, transactions):
        raise NotImplementedError('Subclass responsibility')


class NullMarketExplorer(MarketExplorer):
    @classmethod
    def code(cls):
        return 'null'

    def explore_for(self, estimator, model, transactions):
        return model.ranked_lists[0]


class MIPMarketExplorer(MarketExplorer):
    @classmethod
    def code(cls):
        return 'mip'

    def explore_for(self, estimator, model, transactions):
        problem = MIPMarketExploreLinearProblem(model, transactions)
        final_solutions = LinearSolver().solve(problem, estimator.profiler())

        ranked_lists = []
        for objective_value, values in final_solutions:
            new_ranked_list = [0] * len(model.products)
            for j in model.products:
                position = sum([values['x_%s_%s' % (i, j)] for i in model.products if i != j])
                new_ranked_list[int(position)] = j
            ranked_lists.append(new_ranked_list)
        return ranked_lists


class MIPMarketExploreLinearProblem(LinearProblem):
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions

    def amount_of_variables(self):
        return len(self.objective_coefficients())

    def objective_coefficients(self):
        coefficients = [0.0 for _ in self.model.products for _ in self.model.products]
        return coefficients + [1 / self.model.probability_of(t) for t in self.transactions]

    def lower_bounds(self):
        lower = [0.0] * self.amount_of_variables()
        return lower

    def upper_bounds(self):
        return [1.0] * self.amount_of_variables()

    def variable_types(self):
        variable_types = 'B' * self.amount_of_variables()
        return variable_types

    def variable_names(self):
        variable_names = ['x_%s_%s' % (i, j) for i in self.model.products for j in self.model.products]
        return variable_names + ['w_%s' % t for t in range(len(self.transactions))]

    def constraints(self):
        return MIPMarketExploreConstraints(self.model, self.transactions).constraints()


class MIPMarketExploreConstraints(object):
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions

    def constraints(self):
        independent_terms = []
        names = []
        senses = []
        linear_expressions = []

        self.products_are_ordered_constraints(independent_terms, names, senses, linear_expressions)
        self.transitivity_in_order_constraints(independent_terms, names, senses, linear_expressions)
        self.no_purchase_cannot_be_preferred_constraint(independent_terms, names, senses, linear_expressions)
        self.purchase_compatibility_constraints(independent_terms, names, senses, linear_expressions)

        return {'independent_terms': independent_terms, 'names': names,
                'senses': senses, 'linear_expressions': linear_expressions}

    def products_are_ordered_constraints(self, independent_terms, names, senses, linear_expressions):
        # i before j or j before i, never both at the same time
        for j in self.model.products:
            for i in range(j + 1, len(self.model.products)):
                independent_terms.append(1.0)
                names.append('%s_before_%s_or_%s_before_%s' % (i, j, j, i))
                senses.append('E')
                linear_expressions.append([['x_%s_%s' % (i, j), 'x_%s_%s' % (j, i)], [1.0, 1.0]])

    def transitivity_in_order_constraints(self, independent_terms, names, senses, linear_expressions):
        # transitivity constraints to ensure linear ordering among three products
        for i in self.model.products:
            for j in self.model.products:
                for l in self.model.products:
                    if len({i, j, l}) == 3:
                        independent_terms.append(2.0)
                        names.append('transitivity_for_%s_%s_%s' % (i, j, l))
                        senses.append('L')
                        linear_expressions.append([['x_%s_%s' % (j, i), 'x_%s_%s' % (i, l), 'x_%s_%s' % (l, j)], [1.0, 1.0, 1.0]])

    def no_purchase_cannot_be_preferred_constraint(self, independent_terms, names, senses, linear_expressions):
        # The no-purchase cannot be the preferred option
        independent_terms.append(1.0)
        names.append('no_purchase_is_not_preferred')
        senses.append('G')
        linear_expressions.append([['x_%s_0' % j for j in range(1, len(self.model.products))], [1.0 for _ in range(1, len(self.model.products))]])

    def purchase_compatibility_constraints(self, independent_terms, names, senses, linear_expressions):
        # If I want a profile compatible with transaction t,
        # then all products that are offered must come after the purchased product.
        for t, transaction in enumerate(self.transactions):
            for i in transaction.offered_products:
                if i != transaction.product:
                    independent_terms.append(0.0)
                    names.append(
                        'product_%s_is_worse_than_purchased_if_type_is_compatible_with_transaction_%s' % (i, t))
                    senses += 'L'
                    linear_expressions.append([['w_%s' % t, 'x_%s_%s' % (transaction.product, i)], [1.0, -1.0]])
