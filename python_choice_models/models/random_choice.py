# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from python_choice_models.models import Model


class RandomChoiceModel(Model):
    @classmethod
    def simple_random(cls, products):
        return cls(products)

    @classmethod
    def simple_deterministic(cls, products):
        return cls(products)

    @classmethod
    def code(cls):
        return 'rnd'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'])

    def constraints(self):
        pass

    def data(self):
        return {
            'code': self.code(),
            'products': self.products
        }

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0.0
        return 1.0 / float(len(transaction.offered_products))
