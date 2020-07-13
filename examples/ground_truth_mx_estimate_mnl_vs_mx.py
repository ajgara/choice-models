import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../src/')

from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from settings import Settings
from models import MixedLogitModel, MultinomialLogitModel, RandomChoiceModel
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
    ground_truth = MixedLogitModel(products=products, mus=[p + 1 for p in products], sigmas=[1 if p % 2 == 0 else 10 for p in products])

    offersets = OfferedProductsGenerator(products).generate(TRANSACTIONS)
    transactions = TransactionGenerator(ground_truth).generate_for(offersets)

    initial_solution = MixedLogitModel.simple_deterministic(products)
    estimator_mx = MaximumLikelihoodEstimator()
    estimation_mx = estimator_mx.estimate(initial_solution, transactions)

    initial_solution = MultinomialLogitModel.simple_deterministic(products)
    estimator_mnl = MaximumLikelihoodEstimator()
    estimation_mnl = estimator_mnl.estimate(initial_solution, transactions)

    plot(ground_truth, estimation_mx, estimator_mx, estimation_mnl, estimator_mnl, transactions)


def plot(ground_truth, estimation_mx, estimator_mx, estimation_mnl, estimator_mnl, transactions):
    print('')
    print('Log-likelihood RND: %.4f' % RandomChoiceModel.simple_random(ground_truth.products).log_likelihood_for(transactions))
    print('Log-Likelihood MX: %.4f' % estimation_mx.log_likelihood_for(transactions))
    print('Log-Likelihood MNL: %.4f' % estimation_mnl.log_likelihood_for(transactions))
    print('')
    print('Soft RMSE RND: %.4f' % ground_truth.soft_rmse_for(RandomChoiceModel.simple_random(ground_truth.products)))
    print('Soft RMSE MX: %.4f' % ground_truth.soft_rmse_for(estimation_mx))
    print('Soft RMSE MNL: %.4f' % ground_truth.soft_rmse_for(estimation_mnl))
    print('')

    iterations_mx = estimator_mx.profiler().iterations()
    iterations_mnl = estimator_mnl.profiler().iterations()

    y_data_mx = [-i.value() for i in iterations_mx]
    x_data_mx = [i.stop_time() - iterations_mx[0].start_time() for i in iterations_mx]

    y_data_mnl = [-i.value() for i in iterations_mnl]
    x_data_mnl = [i.stop_time() - iterations_mnl[0].start_time() for i in iterations_mnl]

    plt.plot(x_data_mx, y_data_mx, x_data_mnl, y_data_mnl)

    plt.title('MX vs MNL (Ground truth MX)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log-likelihood')
    plt.legend(['mx', 'mnl'])
    plt.savefig('output/MX vs MNL (Ground truth MX).png')


if __name__ == '__main__':
    main()
