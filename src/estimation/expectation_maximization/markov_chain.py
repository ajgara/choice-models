# This code is from the paper:
# Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.

from numpy.linalg import linalg
from estimation.expectation_maximization import ExpectationMaximizationEstimator


class MarkovChainExpectationMaximizationEstimator(ExpectationMaximizationEstimator):
    def one_step(self, model, transactions):
        X, F = self.expectation_step(model, transactions)
        return self.maximization_step(model, X, F)

    def expectation_step(self, model, transactions):
        # Precalculate psis and thetas.
        psis = []
        thetas = []
        for transaction in transactions:
            psis.append(self.compute_psi(model, transaction))
            thetas.append(model.expected_number_of_visits_if(transaction.offered_products))

        F = self.estimate_F(model, transactions, psis, thetas)
        X = self.estimate_X(model, transactions, psis, thetas)
        return X, F

    def maximization_step(self, model, X, F):
        new_l = []
        new_p = []

        l_denominator = sum([sum(F_t) for F_t in F])
        for product_i in model.products:
            l_numerator = sum([F_t[product_i] for F_t in F])
            new_l.append(l_numerator / l_denominator)

            row = []
            p_denominator = sum([sum(X_t[product_i]) for X_t in X])
            for product_j in model.products:
                p_numerator = sum([X_t[product_i][product_j] for X_t in X])
                if p_denominator:
                    row.append(p_numerator / p_denominator)
                else:
                    row.append(0)
            new_p.append(row)

        model.set_lambdas(new_l)
        model.set_ros(new_p)

        return model

    def compute_psi(self, model, transaction):
        # Calculate P{Z_k (S) = 1 | F_i = 1} for all i. Uses bayes theorem
        not_offered_products = [p for p in model.products if p not in transaction.offered_products]
        A = []
        b = []
        for wanted_product in not_offered_products:
            row = []
            for transition_product in not_offered_products:
                if wanted_product == transition_product:
                    row.append(1.0)
                else:
                    row.append(-model.ro_for(wanted_product, transition_product))
            A.append(row)
            b.append(model.ro_for(wanted_product, transaction.product))

        x = list(linalg.solve(A, b)) if len(A) and len(b) else []  # Maybe all products are offered.

        psi = [0.0 if product in transaction.offered_products else x.pop(0) for product in model.products]
        psi[transaction.product] = 1.0

        return psi

    def estimate_F(self, model, transactions, psis, thetas):
        F = []
        for psi, theta, transaction in zip(psis, thetas, transactions):
            F_t = []
            for product in model.products:
                F_t.append((psi[product] * model.lambda_for(product)) / theta[transaction.product])
            F.append(F_t)
        return F

    def estimate_X(self, model, transactions, psis, thetas):
        X = []
        for psi, theta, transaction in zip(psis, thetas, transactions):
            X_t = []
            for from_product_i in model.products:
                X_t_row = []
                for to_product_j in model.products:
                    if from_product_i in transaction.offered_products:
                        X_t_row.append(0)
                    else:
                        X_t_row.append((psi[to_product_j] * model.ro_for(from_product_i, to_product_j) * theta[from_product_i]) / theta[transaction.product])
                X_t.append(X_t_row)
            X.append(X_t)
        return X
