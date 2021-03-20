# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from python_choice_models.profiler import Profiler


class Estimator(object):
    """
        Estimates a model parameters based on historical transactions data.
    """
    def __init__(self):
        self._profiler = Profiler()

    def profiler(self):
        return self._profiler

    def estimate(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')
