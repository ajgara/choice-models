# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from estimation.maximum_likelihood import MaximumLikelihoodEstimator
from estimation.ranked_list import RankedListEstimator


class RankedListMaximumLikelihoodEstimator(MaximumLikelihoodEstimator, RankedListEstimator):
    pass
