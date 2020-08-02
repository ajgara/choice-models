# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.



class MarketExplorer(object):
    def explore_for(self, estimator, model, transactions):
        raise NotImplementedError('Subclass responsibility')
