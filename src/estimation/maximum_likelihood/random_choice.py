from estimation.maximum_likelihood import MaximumLikelihoodEstimator


class RandomChoiceModelMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def estimate(self, model, transactions):
        return model
