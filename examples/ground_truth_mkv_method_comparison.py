# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../src/')

from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from estimation.expectation_maximization.markov_chain import MarkovChainExpectationMaximizationEstimator
from settings import Settings
from models import MarkovChainModel, MarkovChainRank2Model, RandomChoiceModel
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
    ground_truth = MarkovChainModel.simple_random(products=products)

    offersets = OfferedProductsGenerator(products).generate(TRANSACTIONS)
    transactions = TransactionGenerator(ground_truth).generate_for(offersets)

    initial_solution = MarkovChainModel.simple_deterministic(products)
    estimator_max = MaximumLikelihoodEstimator()
    estimation_max = estimator_max.estimate(initial_solution, transactions)

    initial_solution = MarkovChainModel.simple_deterministic(products)
    estimator_em = MarkovChainExpectationMaximizationEstimator()
    estimation_em = estimator_em.estimate(initial_solution, transactions)

    initial_solution = MarkovChainRank2Model.simple_deterministic(products)
    estimator_mkv2 = MaximumLikelihoodEstimator()
    estimation_mkv2 = estimator_mkv2.estimate(initial_solution, transactions)

    plot(ground_truth, estimator_max, estimator_em, estimator_mkv2, estimation_max, estimation_em, estimation_mkv2, transactions)


def plot(ground_truth, estimator_max, estimator_em, estimator_mkv2, estimation_max, estimation_em, estimation_mkv2, transactions):
    print('')
    print('Log-likelihood RND: %.4f' % RandomChoiceModel.simple_random(ground_truth.products).log_likelihood_for(transactions))
    print('Log-likelihood MAX: %.4f' % estimation_max.log_likelihood_for(transactions))
    print('Log-likelihood EM: %.4f' % estimation_em.log_likelihood_for(transactions))
    print('Log-likelihood R2: %.4f' % estimation_mkv2.log_likelihood_for(transactions))
    print('')
    print('Soft RMSE RND: %.4f' % ground_truth.soft_rmse_for(RandomChoiceModel.simple_random(ground_truth.products)))
    print('Soft RMSE MAX: %.4f' % ground_truth.soft_rmse_for(estimation_max))
    print('Soft RMSE EM: %.4f' % ground_truth.soft_rmse_for(estimation_em))
    print('Soft RMSE R2: %.4f' % ground_truth.soft_rmse_for(estimation_mkv2))
    print('')

    iterations_max = estimator_max.profiler().iterations()[2:]
    iterations_em = estimator_em.profiler().iterations()[2:]
    iterations_mkv2 = estimator_mkv2.profiler().iterations()[2:]

    y_mkv2_data = [-i.value() for i in iterations_mkv2]
    x_mkv2_data = [i.stop_time() - iterations_mkv2[0].start_time() for i in iterations_mkv2]

    y_max_data = [-i.value() for i in iterations_max]
    x_max_data = [i.stop_time() - iterations_max[0].start_time() for i in iterations_max]

    y_em_data = [i.value() for i in iterations_em]
    x_em_data = [i.stop_time() - iterations_em[0].start_time() for i in iterations_em]

    plt.plot(x_max_data, y_max_data, x_em_data, y_em_data, x_mkv2_data, y_mkv2_data)
    plt.title('Markov Chain method comparison')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log-likelihood')
    plt.legend(['max', 'em', 'rank2'])
    plt.savefig('output/Markov Chain method comparison.png')


if __name__ == '__main__':
    main()
