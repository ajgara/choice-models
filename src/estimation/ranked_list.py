# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from copy import deepcopy

from scipy.stats import chi2

from estimation import Estimator
from estimation.market_explore.ranked_list import NullMarketExplorer


class RankedListEstimator(Estimator):
    @classmethod
    def with_this(cls, market_explorer):
        return cls(market_explorer)

    def __init__(self, market_explorer=NullMarketExplorer()):
        super(RankedListEstimator, self).__init__()
        self.market_explorer = market_explorer

    def estimate(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')

    def estimate_with_market_discovery(self, model, transactions):
        model = self.estimate(model, transactions)
        new_ranked_lists = self.market_explorer.explore_for(self, model, transactions)
        add, new_model = self.is_worth_adding(model, new_ranked_lists, transactions)

        while add and self.profiler().duration() < 1800.0:
            model = new_model
            print(('Adding list... amount of lists %s' % len(model.all_betas())))
            new_ranked_lists = self.market_explorer.explore_for(self, model, transactions)
            add, new_model = self.is_worth_adding(model, new_ranked_lists, transactions)

        return model

    def is_worth_adding(self, model, new_ranked_lists, transactions):
        new_model = deepcopy(model)
        for new_ranked_list in new_ranked_lists:
            new_model.add_ranked_list(new_ranked_list)
        new_model = self.estimate(new_model, transactions)

        return self.compare_statistical_significance(model, new_model, transactions), new_model

    def compare_statistical_significance(self, model, new_model, transactions):
        likelihood_1 = model.log_likelihood_for(transactions)
        likelihood_2 = new_model.log_likelihood_for(transactions)
        likelihood_ratio = -2.0 * (likelihood_1 - likelihood_2) * len(transactions)
        dimensionality_difference = len(new_model.parameters_vector()) - len(model.parameters_vector())
        return likelihood_ratio > chi2.isf(q=0.05, df=dimensionality_difference)
