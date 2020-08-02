# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

import random
import numpy

from utils import powerset


class Transaction(object):
    @classmethod
    def from_json(cls, json_list):
        return [cls(d['product'], d['offered_products']) for d in json_list]

    @classmethod
    def all_for(cls, model):
        products = set(model.products) - {0}
        for offer_set in powerset(products):
            for product in [0] + sorted(offer_set):
                yield cls(product, [0] + sorted(offer_set))

    def __init__(self, product, offered_products):
        self.product = product
        self.offered_products = offered_products

    def as_json(self):
        return {'product': self.product, 'offered_products': self.offered_products}

    def __hash__(self):
        return (self.product, tuple(self.offered_products)).__hash__()

    def __eq__(self, other):
        return self.product == other.product and self.offered_products == other.offered_products

    def __repr__(self):
        return "<Product: %s ; Offered products: %s >" % (self.product, self.offered_products)


class TransactionGenerator(object):
    def __init__(self, model):
        self.model = model

    def generate_for(self, lists_of_offered_products):
        transactions = []
        for i, offered_products in enumerate(lists_of_offered_products):
            transactions.append(self.generate_transaction_for(offered_products))
        return transactions

    def generate_transaction_for(self, offered_products):
        distribution = self.model.probability_distribution_over(offered_products)
        purchased_product = list(numpy.random.multinomial(1, distribution, 1)[0]).index(1)
        return Transaction(purchased_product, offered_products)


class OfferedProductsGenerator(object):
    def __init__(self, products):
        self.products = products

    def generate_distinct(self, amount, min_times_offered=1, max_times_offered=1):
        offer_sets = []
        while len(offer_sets) < amount:
            offered_products = self.generate_offered_products()
            if offered_products != [0] and offered_products not in offer_sets:
                amount_of_times_offered = random.choice(list(range(min_times_offered, max_times_offered + 1)))
                for i in range(amount_of_times_offered):
                    offer_sets.append(offered_products)
        random.shuffle(offer_sets)
        return offer_sets

    def generate(self, amount, offering_probability=0.5):
        offer_sets = []
        while len(offer_sets) < amount:
            offer_sets.append(self.generate_offered_products(offering_probability))
        return offer_sets

    def generate_with_size(self, amount, minimum, maximum):
        offer_sets = []
        while len(offer_sets) < amount:
            offer_sets.append(self.generate_offer_set_with_size(minimum, maximum))
        return offer_sets

    def generate_offer_set_with_size(self, minimum, maximum):
        offered = {0}
        size = random.choice(list(range(minimum, maximum + 1)))
        while len(offered) < size + 1:
            offered.add(random.choice(self.products))
        return sorted(offered)

    def generate_offered_products(self, offering_probability=0.5):
        offered_products = [0]
        for i in range(1, len(self.products)):
            is_offered = random.uniform(0, 1) < offering_probability
            if is_offered:
                offered_products.append(i)
        return offered_products


class RankedListGenerator(object):
    def __init__(self, products):
        self.products = products

    def generate(self, amount_ranked_lists):
        ranked_lists = self.singletons()

        while len(ranked_lists) < amount_ranked_lists + len(self.products):
            ranked_list = list(numpy.random.permutation(len(self.products)))
            if ranked_list not in ranked_lists:
                ranked_lists.append(ranked_list)

        return ranked_lists

    def singletons(self):
        singletons = []
        all_products = set(self.products)

        for first_product in all_products:
            ranked_list = [first_product] + numpy.random.permutation(list(all_products - {first_product}))
            singletons.append(ranked_list)
        return singletons
