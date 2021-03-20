# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from python_choice_models.settings import Settings
import json
from python_choice_models.models import Model, ExponomialModel, MultinomialLogitModel, RandomChoiceModel, LatentClassModel, MarkovChainModel, \
    MarkovChainRank2Model, MixedLogitModel, NestedLogitModel, RankedListModel
from python_choice_models.estimation.maximum_likelihood import MaximumLikelihoodEstimator
from python_choice_models.estimation.maximum_likelihood.latent_class import LatentClassFrankWolfeEstimator
from python_choice_models.estimation.maximum_likelihood.random_choice import RandomChoiceModelMaximumLikelihoodEstimator
from python_choice_models.estimation.expectation_maximization.markov_chain import MarkovChainExpectationMaximizationEstimator
from python_choice_models.estimation.expectation_maximization.ranked_list import RankedListExpectationMaximizationEstimator
from python_choice_models.estimation.market_explore.ranked_list import MIPMarketExplorer

from python_choice_models.transactions.base import Transaction
import numpy as np
import random
import matplotlib.pyplot as plt


random.seed(1)
np.random.seed(1)


TRANSACTIONS = 3000


def read_synthetic_instance(file_name):
    with open(file_name, 'r') as f:
        data = json.loads(f.read())
    ground_truth = Model.from_data(data['ground_model'])
    transactions = Transaction.from_json(data['transactions']['in_sample_transactions'])
    return ground_truth, transactions


def main():
    Settings.new(
        linear_solver_partial_time_limit=300.0,
        non_linear_solver_partial_time_limit=300.0,
        solver_total_time_limit=1800.0,
    )

    ground_truth, transactions = read_synthetic_instance('instances/10_lists_300_periods_1_instance.json')
    products = ground_truth.products

    nests = [{'products': [0], 'lambda': 0.8},
             {'products': [i for i in range(1, len(products)) if i % 2 == 0], 'lambda': 0.8},
             {'products': [i for i in range(1, len(products)) if i % 2 == 1], 'lambda': 0.8}]

    lists = [[i] + list(sorted(set(products) - {i})) for i in range(len(products))]

    models_to_run = [
        (ExponomialModel.simple_deterministic(products), MaximumLikelihoodEstimator()),
        (MultinomialLogitModel.simple_deterministic(products), MaximumLikelihoodEstimator()),
        (RandomChoiceModel.simple_deterministic(products), RandomChoiceModelMaximumLikelihoodEstimator()),
        (LatentClassModel.simple_deterministic(products, 1), LatentClassFrankWolfeEstimator()),
        (MarkovChainModel.simple_deterministic(products), MarkovChainExpectationMaximizationEstimator()),
        (MarkovChainRank2Model.simple_deterministic(products), MaximumLikelihoodEstimator()),
        (MixedLogitModel.simple_deterministic(products), MaximumLikelihoodEstimator()),
        (NestedLogitModel.simple_deterministic(products, nests=nests), MaximumLikelihoodEstimator()),
        (RankedListModel.simple_deterministic(products, ranked_lists=lists), RankedListExpectationMaximizationEstimator(MIPMarketExplorer())),
    ]

    results = []
    for initial_solution, estimator in models_to_run:
        if hasattr(estimator, 'estimate_with_market_discovery'):
            model = estimator.estimate_with_market_discovery(initial_solution, transactions)
        else:
            model = estimator.estimate(initial_solution, transactions)
        soft_rmse = model.soft_rmse_for(ground_truth)
        results.append((model, estimator.profiler(), soft_rmse))

    plot(results)


def plot(results):
    cells = []
    rows = []
    print('')
    for model, profile, soft_rmse in sorted(results, key=lambda x: x[2]):
        print('SoftRMSE %s: %.4f' % (model.__class__.__name__, soft_rmse))
        cells.append(['%.4f' % soft_rmse])
        rows.append(model.__class__.__name__)
    print('')

    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    axs.axis('tight')
    axs.axis('off')
    axs.table(cellText=cells, rowLabels=rows, colLabels=['SoftRMSE'], loc='center', colWidths=[0.3, 0.3])
    plt.savefig('output/10 lists instance results.png')


if __name__ == '__main__':
    main()
