# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../src/')

from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from estimation.maximum_likelihood.latent_class import LatentClassFrankWolfeEstimator
from estimation.expectation_maximization.latent_class import LatentClassExpectationMaximizationEstimator
from settings import Settings
from models import LatentClassModel, RandomChoiceModel, RankedListModel
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
        solver_total_time_limit=1800,
    )

    products = [i for i in range(11)]
    number_lists = 10
    lists = [list(np.random.permutation(len(products))) for _ in range(number_lists)]
    ground_truth = RankedListModel.simple_random(products=products, ranked_lists=lists)

    offersets = OfferedProductsGenerator(products).generate(TRANSACTIONS)
    transactions = TransactionGenerator(ground_truth).generate_for(offersets)

    initial_solution = LatentClassModel.simple_deterministic(products, 10)
    estimator_max = MaximumLikelihoodEstimator()
    estimation_max = estimator_max.estimate(initial_solution, transactions)

    initial_solution = LatentClassModel.simple_deterministic(products, 10)
    estimator_em = LatentClassExpectationMaximizationEstimator()
    estimation_em = estimator_em.estimate(initial_solution, transactions)

    initial_solution = LatentClassModel.simple_deterministic(products, 10)
    estimator_fw = LatentClassFrankWolfeEstimator()
    estimation_fw = estimator_fw.estimate(initial_solution, transactions)

    plot(ground_truth, estimation_em, estimation_max, estimation_fw, estimator_em, estimator_max, estimator_fw, transactions)


def plot(ground_truth, estimation_em, estimation_max, estimation_fw, estimator_em, estimator_max, estimator_fw, transactions):
    print('')
    print('Log-likelihood RND: %.4f' % RandomChoiceModel.simple_random(ground_truth.products).log_likelihood_for(transactions))
    print('Log-likelihood MAX: %.4f' % estimation_max.log_likelihood_for(transactions))
    print('Log-likelihood EM: %.4f' % estimation_em.log_likelihood_for(transactions))
    print('Log-likelihood FW: %.4f' % estimation_fw.log_likelihood_for(transactions))
    print('')
    print('Soft RMSE RND: %.4f' % ground_truth.soft_rmse_for(RandomChoiceModel.simple_random(ground_truth.products)))
    print('Soft RMSE MAX: %.4f' % ground_truth.soft_rmse_for(estimation_max))
    print('Soft RMSE EM: %.4f' % ground_truth.soft_rmse_for(estimation_em))
    print('Soft RMSE FW: %.4f' % ground_truth.soft_rmse_for(estimation_fw))
    print('')

    iterations_fw = estimator_fw.profiler().iterations()
    iterations_max = estimator_max.profiler().iterations()
    iterations_em = estimator_em.profiler().iterations()

    # Frank wolfe has two subproblems on each iteration.
    # - 1st problem finds new MNLs to add to the Latent class: (always negative objective function)
    # - 2nd problem corrects weights for each class (always positive objective function, basically a NLL)

    # Not very elegant:
    # - Plot only NLL after each iteration.

    nlls_iterations = []
    for i in range(1, len(iterations_fw)):
        # If went from positive to negative, then changed from 2nd problem to 1st problem. Save NLL.
        if iterations_fw[i - 1].value() >= 0 and iterations_fw[i].value() < 0:
            nlls_iterations.append(
                (-iterations_fw[i - 1].value(), iterations_fw[i - 1].stop_time() - iterations_fw[0].start_time()))
    if iterations_fw[-1].value() >= 0:
        nlls_iterations.append(
            (-iterations_fw[-1].value(), iterations_fw[-1].stop_time() - iterations_fw[0].start_time()))

    y_data_fw = [x[0] * len(set(transactions)) / len(transactions) for x in nlls_iterations]
    x_data_fw = [x[1] for x in nlls_iterations]

    y_data_max = [-i.value() for i in iterations_max]
    x_data_max = [i.stop_time() - iterations_max[0].start_time() for i in iterations_max]

    y_data_em = [i.value() for i in iterations_em]
    x_data_em = [i.stop_time() - iterations_em[0].start_time() for i in iterations_em]

    plt.plot(x_data_fw, y_data_fw, x_data_max, y_data_max, x_data_em, y_data_em)
    plt.title('Latent Class method comparison')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Log-likelihood')
    plt.legend(['CG (%s classes discovered)' % len(estimation_fw.gammas),
                'MAX (%s classes given)' % len(estimation_max.gammas),
                'EM (%s classes given)' % len(estimation_em.gammas)])
    plt.savefig('output/Latent Class method comparison.png')


if __name__ == '__main__':
    main()
