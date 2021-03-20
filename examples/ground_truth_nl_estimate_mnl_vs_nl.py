# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from python_choice_models.estimation.maximum_likelihood import MaximumLikelihoodEstimator
from python_choice_models.settings import Settings
from python_choice_models.models import NestedLogitModel, MultinomialLogitModel, RandomChoiceModel
from python_choice_models.transactions.base import TransactionGenerator, OfferedProductsGenerator
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
    nests = [
        {'lambda': 0.8, 'products': [0, 1, 2]},
        {'lambda': 0.8, 'products': [3, 4, 5]},
        {'lambda': 0.8, 'products': [6, 7, 8]},
        {'lambda': 0.8, 'products': [9, 10]},
    ]

    products = [i for i in range(11)]
    ground_truth = NestedLogitModel(products=products, etas=[0.1 * i for i in range(1, 12)], nests=nests)

    offersets = OfferedProductsGenerator(products).generate(TRANSACTIONS)
    transactions = TransactionGenerator(ground_truth).generate_for(offersets)

    initial_solution = NestedLogitModel.simple_deterministic(products, nests)
    estimator_nl = MaximumLikelihoodEstimator()
    estimation_nl = estimator_nl.estimate(initial_solution, transactions)

    initial_solution = MultinomialLogitModel.simple_deterministic(products)
    estimator_mnl = MaximumLikelihoodEstimator()
    estimation_mnl = estimator_mnl.estimate(initial_solution, transactions)

    plot(ground_truth, estimator_mnl, estimation_mnl, estimator_nl, estimation_nl, transactions)


def plot(ground_truth, estimator_mnl, estimation_mnl, estimator_nl, estimation_nl, transactions):
    print('')
    print('Log-likelihood RND: %.4f' % RandomChoiceModel.simple_random(ground_truth.products).log_likelihood_for(transactions))
    print('Log-Likelihood MNL: %.4f' % estimation_mnl.log_likelihood_for(transactions))
    print('Log-Likelihood NL: %.4f' % estimation_nl.log_likelihood_for(transactions))
    print('')
    print('Soft RMSE RND: %.4f' % ground_truth.soft_rmse_for(RandomChoiceModel.simple_random(ground_truth.products)))
    print('Soft RMSE MNL: %.4f' % ground_truth.soft_rmse_for(estimation_mnl))
    print('Soft RMSE NL: %.4f' % ground_truth.soft_rmse_for(estimation_nl))
    print('')

    iterations_nl = estimator_nl.profiler().iterations()
    iterations_mnl = estimator_mnl.profiler().iterations()

    y_data_nl = [-i.value() for i in iterations_nl]
    x_data_nl = [i.stop_time() - iterations_nl[0].start_time() for i in iterations_nl]

    y_data_mnl = [-i.value() for i in iterations_mnl]
    x_data_mnl = [i.stop_time() - iterations_mnl[0].start_time() for i in iterations_mnl]

    plt.plot(x_data_nl, y_data_nl, x_data_mnl, y_data_mnl)

    plt.title('NL vs MNL (Ground truth NL)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log-likelihood')
    plt.legend(['nl', 'mnl'])
    plt.savefig('output/NL vs MNL (Ground truth NL).png')


if __name__ == '__main__':
    main()
