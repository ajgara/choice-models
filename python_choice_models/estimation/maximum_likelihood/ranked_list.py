# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from python_choice_models.estimation.maximum_likelihood import MaximumLikelihoodEstimator
from python_choice_models.estimation.ranked_list import RankedListEstimator


class RankedListMaximumLikelihoodEstimator(MaximumLikelihoodEstimator, RankedListEstimator):
    pass
