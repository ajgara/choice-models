from profiler import Profiler


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
