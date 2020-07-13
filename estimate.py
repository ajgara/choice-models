# Electronic companion of the paper:
# "A Comparative Empirical Study of Discrete Choice Models in Retail Operations" by
# - Gerardo Berbeglia: Melbourne Business School, University of Melbourne, g.berbeglia@mbs.edu
# - Agustin Garassino: School of Sciences, Universidad de Buenos Aires, and School of Business, Universidad Torcuato di Tella, Buenos Aires, Argentina, agarassino@dc.uba.ar
# - Gustavo Vulcano: School of Business, Universidad Torcuato di Tella, and CONICET, Buenos Aires, Argentina, gvulcano@utdt.edu

# This code was written in Python and tested on Ubunutu Linux 17.04 operating system. 
# It is designed with focus on its reusability and maintainability, taking advantage of several object oriented programming (OOP) features of Python. 
# The implementation of the different algorithms is intended to be self-explanatory in terms of the names of the objects and functions used along the way.

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/src/')

from estimation.market_explore.ranked_list import MIPMarketExplorer
from estimation.expectation_maximization.markov_chain import MarkovChainExpectationMaximizationEstimator
from estimation.expectation_maximization.ranked_list import RankedListExpectationMaximizationEstimator
from estimation.expectation_maximization.latent_class import LatentClassExpectationMaximizationEstimator
from estimation.maximum_likelihood.random_choice import RandomChoiceModelMaximumLikelihoodEstimator
from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from estimation.maximum_likelihood.latent_class import LatentClassFrankWolfeEstimator
from estimation.maximum_likelihood.ranked_list import RankedListMaximumLikelihoodEstimator

from settings import Settings

from models import Model, MixedLogitModel, MultinomialLogitModel, ExponomialModel, LatentClassModel, MarkovChainModel, MarkovChainRank2Model, NestedLogitModel, RandomChoiceModel, RankedListModel

from transactions.base import Transaction

GLOBAL_TIME_LIMIT = 1800

NORMAL_SETTINGS = {
    'linear_solver_partial_time_limit': None,
    'non_linear_solver_partial_time_limit': None,
    'solver_total_time_limit': 1800.0,
}

RANKED_LIST_SETTINGS = {
    'linear_solver_partial_time_limit': 300,
    'non_linear_solver_partial_time_limit': 300,
    'solver_total_time_limit': 1800.0,
}

LATENT_CLASS_SETTINGS = {
    'linear_solver_partial_time_limit': None,
    'non_linear_solver_partial_time_limit': 300,
    'solver_total_time_limit': 1800.0,
}

estimators = {
    'max': {
        'name': 'Standard Maximum Likelihood',
        'models': {
            'exp': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: ExponomialModel.simple_deterministic(products),
                'name': 'Exponomial',
                'settings': NORMAL_SETTINGS
            },
            'mkv': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: MarkovChainModel.simple_deterministic(products),
                'name': 'Markov Chain',
                'settings': NORMAL_SETTINGS
            },
            'mkv2': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: MarkovChainRank2Model.simple_deterministic(products),
                'name': 'Markov Chain Rank 2',
                'settings': NORMAL_SETTINGS
            },
            'mnl': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: MultinomialLogitModel.simple_deterministic(products),
                'name': 'Multinomial Logit',
                'settings': NORMAL_SETTINGS
            },
            'nl': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: NestedLogitModel.simple_deterministic_ordered_nests(products, [1, len(products) // 2, len(products) - (len(products) // 2) - 1]),
                'name': 'Nested Logit',
                'settings': NORMAL_SETTINGS
            },
            'mx': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: MixedLogitModel.simple_deterministic(products),
                'name': 'Mixed Logit',
                'settings': NORMAL_SETTINGS
            },
            'rnd': {
                'estimator': RandomChoiceModelMaximumLikelihoodEstimator(),
                'model_class': lambda products: RandomChoiceModel.simple_deterministic(products),
                'name': 'Random Choice',
                'settings': NORMAL_SETTINGS
            },
            'rl': {
                'estimator': RankedListMaximumLikelihoodEstimator.with_this(MIPMarketExplorer()),
                'model_class': lambda products: RankedListModel.simple_deterministic_independent(products),
                'name': 'Ranked List',
                'settings': RANKED_LIST_SETTINGS
            },
            'lc': {
                'estimator': MaximumLikelihoodEstimator(),
                'model_class': lambda products: LatentClassModel.simple_deterministic(products, 5),
                'name': 'Latent Class',
                'settings': RANKED_LIST_SETTINGS
            }
        }
    },
    'em': {
        'name': 'Expectation Maximization',
        'models': {
            'mkv': {
                'estimator': MarkovChainExpectationMaximizationEstimator(),
                'model_class': lambda products: MarkovChainModel.simple_deterministic(products),
                'name': 'Markov Chain',
                'settings': NORMAL_SETTINGS
            },
            'rl': {
                'estimator': RankedListExpectationMaximizationEstimator.with_this(MIPMarketExplorer()),
                'model_class': lambda products: RankedListModel.simple_deterministic_independent(products),
                'name': 'Ranked List',
                'settings': RANKED_LIST_SETTINGS
            },
            'lc': {
                'estimator': LatentClassExpectationMaximizationEstimator(),
                'model_class': lambda products: LatentClassModel.simple_deterministic(products, 5),
                'name': 'Latent Class',
                'settings': RANKED_LIST_SETTINGS
            }
        }
    },
    'fw': {
        'name': 'Frank Wolfe/Conditional Gradient',
        'models': {
            'lc': {
                'estimator': LatentClassFrankWolfeEstimator(),
                'model_class': lambda products: LatentClassModel.simple_deterministic(products, 1),
                'name': 'Latent Class',
                'settings': NORMAL_SETTINGS
            },
        }
    }
}


def read_synthetic_instance(file_name):
    with open(file_name, 'r') as f:
        data = json.loads(f.read())
    ground_truth = Model.from_data(data['ground_model'])
    transactions = Transaction.from_json(data['transactions']['in_sample_transactions'])
    return ground_truth, transactions


def print_usage():
    print('Use:')
    print(' - Run just one model: python estimate.py <estimation_method_code> <model_code> <input_file> <output_file>')
    print(' - Example: python estimate.py max mnl examples/instances/30_periods_1_instance.json mnl_estimation.json')
    print('')

    print('Estimation methods codes:')
    for estimation_method, method_info in estimators.items():
        print(' - (%s) %s' % (estimation_method, method_info['name']))
    print('')

    print('Model codes and compatibilities with estimation methods:')
    for estimation_method, method_info in sorted(estimators.items(), key=lambda x: x[0]):
        print(' - (%s) %s' % (estimation_method, method_info['name']))
        for model, model_info in sorted(method_info['models'].items(), key=lambda x: x[0]):
            print('    - (%s) %s' % (model, model_info['name']))
    print('')


def run(estimation_method, model, input_file, output_file):
    print(' * Reading input file...')
    ground_truth, transactions = read_synthetic_instance(input_file)
    products = ground_truth.products

    print('    - Amount of transactions: %s' % len(transactions))
    print('    - Amount of products: %s' % len(products))

    print(' * Retrieving estimation method...')
    model_info = estimators[estimation_method]['models'][model]

    print(' * Retrieving settings...')
    Settings.new(
        linear_solver_partial_time_limit=model_info['settings']['linear_solver_partial_time_limit'],
        non_linear_solver_partial_time_limit=model_info['settings']['non_linear_solver_partial_time_limit'],
        solver_total_time_limit=model_info['settings']['solver_total_time_limit'],
    )

    print(' * Creating initial solution...')
    model = model_info['model_class'](products)

    print(' * Starting estimation...')
    if hasattr(model_info['estimator'], 'estimate_with_market_discovery'):
        result = model_info['estimator'].estimate_with_market_discovery(model, transactions)
    else:
        result = model_info['estimator'].estimate(model, transactions)

    rmse = ground_truth.soft_rmse_for(result)

    print(' * Computing result...')
    print('')
    print('Soft RMSE: %s' % rmse)
    print('')

    result.save(output_file)
    return rmse


def main():
    try:
        if len(sys.argv) == 5:
            output_file = sys.argv[4]
            input_file = sys.argv[3]
            model = sys.argv[2]
            estimation_method = sys.argv[1]
            run(estimation_method, model, input_file, output_file)
        else:
            print('Wrong number of arguments!\n')
            print_usage()
    except KeyError as error:
        print('Estimator not found!\n')
        print_usage()


main()
