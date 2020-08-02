# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../src/')

from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from settings import Settings
from models import ExponomialModel, MultinomialLogitModel, RandomChoiceModel
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
    ground_truth = ExponomialModel.simple_random(products=products)

    offersets = OfferedProductsGenerator(products).generate(TRANSACTIONS)
    transactions = TransactionGenerator(ground_truth).generate_for(offersets)

    initial_solution = ExponomialModel.simple_deterministic(products)
    estimator_exp = MaximumLikelihoodEstimator()
    estimation_exp = estimator_exp.estimate(initial_solution, transactions)

    initial_solution = MultinomialLogitModel.simple_deterministic(products)
    estimator_mnl = MaximumLikelihoodEstimator()
    estimation_mnl = estimator_mnl.estimate(initial_solution, transactions)

    plot(ground_truth, estimation_exp, estimator_exp, estimation_mnl, estimator_mnl, transactions)


def plot(ground_truth, estimation_exp, estimator_exp, estimation_mnl, estimator_mnl, transactions):
    print('')
    print('Log-likelihood RND: %.4f' % RandomChoiceModel.simple_random(ground_truth.products).log_likelihood_for(transactions))
    print('Log-likelihood EXP: %.4f' % estimation_exp.log_likelihood_for(transactions))
    print('Log-likelihood MNL: %.4f' % estimation_mnl.log_likelihood_for(transactions))
    print('')
    print('Soft RMSE RND: %.4f' % ground_truth.soft_rmse_for(RandomChoiceModel.simple_random(ground_truth.products)))
    print('Soft RMSE EXP: %.4f' % ground_truth.soft_rmse_for(estimation_exp))
    print('Soft RMSE MNL: %.4f' % ground_truth.soft_rmse_for(estimation_mnl))
    print('')

    iterations_exp = estimator_exp.profiler().iterations()
    iterations_mnl = estimator_mnl.profiler().iterations()

    y_data_exp = [-i.value() for i in iterations_exp]
    x_data_exp = [i.stop_time() - iterations_exp[0].start_time() for i in iterations_exp]

    y_data_mnl = [-i.value() for i in iterations_mnl]
    x_data_mnl = [i.stop_time() - iterations_mnl[0].start_time() for i in iterations_mnl]

    plt.plot(x_data_exp, y_data_exp, x_data_mnl, y_data_mnl)
    plt.title('MNL vs EXP (Ground truth EXP)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log-likelihood')
    plt.legend(['exp', 'mnl'])
    plt.savefig('output/MNL vs EXP (Ground truth EXP).png')


if __name__ == '__main__':
    main()
