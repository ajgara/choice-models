import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../src/')

from estimation.market_explore.ranked_list import MIPMarketExplorer
from estimation.maximum_likelihood.ranked_list import RankedListMaximumLikelihoodEstimator
from estimation.expectation_maximization.ranked_list import RankedListExpectationMaximizationEstimator
from settings import Settings
from models import RankedListModel, RandomChoiceModel
from transactions.base import TransactionGenerator, OfferedProductsGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1)
np.random.seed(1)


TRANSACTIONS = 3000


def main():
    Settings.new(
        linear_solver_partial_time_limit=None,
        non_linear_solver_partial_time_limit=None,
        solver_total_time_limit=20000,
    )

    products = [i for i in range(11)]
    number_lists = 5
    lists = [list(np.random.permutation(len(products))) for _ in range(number_lists)]
    ground_truth = RankedListModel.simple_random(products=products, ranked_lists=lists)

    offersets = OfferedProductsGenerator(products).generate(TRANSACTIONS)
    transactions = TransactionGenerator(ground_truth).generate_for(offersets)

    initial_solution = RankedListModel.simple_deterministic(products, ranked_lists=lists)
    estimator_max = RankedListMaximumLikelihoodEstimator()
    estimation_max = estimator_max.estimate(initial_solution, transactions)

    initial_solution = RankedListModel.simple_deterministic(products, ranked_lists=lists)
    estimator_em = RankedListExpectationMaximizationEstimator()
    estimation_em = estimator_em.estimate(initial_solution, transactions)

    initial_solution = RankedListModel.simple_deterministic_independent(products)
    estimator_mip = RankedListExpectationMaximizationEstimator.with_this(MIPMarketExplorer())
    estimation_mip = estimator_mip.estimate_with_market_discovery(initial_solution, transactions)

    plot(ground_truth, estimator_max, estimator_em, estimator_mip, estimation_max, estimation_em, estimation_mip, transactions)


def plot(ground_truth, estimator_max, estimator_em, estimator_mip, estimation_max, estimation_em, estimation_mip, transactions):
    print('')
    print('Log-likelihood RND: %.4f' % RandomChoiceModel.simple_random(ground_truth.products).log_likelihood_for(transactions))
    print('Log-likelihood MAX: %.4f' % estimation_max.log_likelihood_for(transactions))
    print('Log-likelihood EM: %.4f' % estimation_em.log_likelihood_for(transactions))
    print('Log-likelihood MIP: %.4f' % estimation_mip.log_likelihood_for(transactions))
    print('')
    print('Soft RMSE RND: %.4f' % ground_truth.soft_rmse_for(RandomChoiceModel.simple_random(ground_truth.products)))
    print('Soft RMSE MAX: %.4f' % ground_truth.soft_rmse_for(estimation_max))
    print('Soft RMSE EM: %.4f' % ground_truth.soft_rmse_for(estimation_em))
    print('Soft RMSE MIP: %.4f' % ground_truth.soft_rmse_for(estimation_mip))
    print('')
    iterations_em = estimator_em.profiler().iterations()
    iterations_max = estimator_max.profiler().iterations()
    iterations_mip = estimator_mip.profiler().iterations()

    y_data_max = [-i.value() for i in iterations_max]
    x_data_max = [i.stop_time() - iterations_max[0].start_time() for i in iterations_max]
    y_data_em = [i.value() for i in iterations_em]
    x_data_em = [i.stop_time() - iterations_em[0].start_time() for i in iterations_em]
    y_data_mip = [i.value() for i in iterations_mip]
    x_data_mip = [i.stop_time() - iterations_mip[0].start_time() for i in iterations_mip]

    plt.plot(x_data_max, y_data_max, x_data_em, y_data_em, x_data_mip, y_data_mip)
    plt.title('Ranked List Method comparison')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log-likelihood')
    plt.legend(['max (has lists)', 'em (has lists)', 'mip (discovers lists)'])
    plt.savefig('output/Ranked List Method comparison.png')


if __name__ == '__main__':
    main()
