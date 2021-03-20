# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from python_choice_models.estimation import Estimator
from python_choice_models.settings import Settings
from python_choice_models.utils import time_for_optimization
import time


class ExpectationMaximizationEstimator(Estimator):
    def estimate(self, model, transactions):
        self.profiler().reset_convergence_criteria()
        self.profiler().update_time()
        model = self.custom_initial_solution(model, transactions)
        cpu_time = time_for_optimization(partial_time=Settings.instance().non_linear_solver_partial_time_limit(),
                                         total_time=Settings.instance().solver_total_time_limit(),
                                         profiler=self.profiler())

        start_time = time.time()
        while True:
            self.profiler().start_iteration()
            model = self.one_step(model, transactions)
            likelihood = model.log_likelihood_for(transactions)
            self.profiler().stop_iteration(likelihood)

            if self.profiler().should_stop() or (time.time() - start_time) > cpu_time:
                break

        return model

    def one_step(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')

    def custom_initial_solution(self, model, transactions):
        return model
