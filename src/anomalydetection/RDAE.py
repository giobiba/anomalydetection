import numpy as np
import numpy.linalg as linalg

from DeepAE import DAE


class RDAE:
    """
        Parameters
        ----------
        lambda_:
            penalty for the sparsity error
        mu_:
            initial lagrangian penalty
        max_iter:
            maximum number of iterations
        rho_:
            learning rate
        tau_:
            mu update criterion parameter
        REL_TOL:
            relative tolerence
        ABS_TOL:
            absolute tolerance
    """
    def __init__(self, encoder_neurons=None, decoder_neurons=None, lambda_=1.0, error=1.0e-7,
                 activation_function='tanh', regularizer='l2', regularizer_penalty=0.1, dropout=0.2, shrink='l1shrink',
                 verbose=False, max_iter=100):

        self.history = []
        self.lambda_ = lambda_
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.activation_function = activation_function
        self.regularizer = regularizer
        self.regularizer_penalty = regularizer_penalty
        self.dropout = dropout
        self.error = error
        self.max_iter = max_iter
        self.AE = DAE(encoder_layer_size=self.encoder_neurons, decoder_layer_size=self.decoder_neurons,
                      activation_function=self.activation_function, regularizer=self.regularizer, regularizer_penalty=self.regularizer_penalty,
                      dropout=self.dropout)
        if shrink == 'l1shrink':
            from src.anomalydetection.utils.utils import l1shrink as shrink_function
        elif shrink == 'l21shrink':
            from src.anomalydetection.utils.utils import l21shrink as shrink_function
        else:
            from src.anomalydetection.utils.utils import shrink as shrink_function
        self.shrink_function = shrink_function
        self.verbose = verbose

    def compile(self, **kwargs):
        self.AE.compile(**kwargs)

    def fit(self, x, inner_iteration=50, batch_size=50):

        # check if x shape matches the input of the autoencoder
        assert x.shape[1] == self.encoder_neurons[0]

        self.STATS = {
            'c1': [],
            'c2': []
        }

        self.L = np.zeros(x.shape)
        self.S = np.zeros(x.shape)

        # calculate mu
        mu = x.size / (4.0 * linalg.norm(x, 1))

        print(f"mu: {mu}")
        print(f"shrink parameter: {self.lambda_ / mu}")
        LS0 = self.L + self.S

        XFnorm = linalg.norm(x, 'fro')
        for i in range(self.max_iter):
            if self.verbose:
                print(f"Iteration: {i}")
            self.L = x - self.S

            # Train the autoencoder with L
            self.history.append(self.AE.fit(x=self.L, y=self.L,
                                       epochs=inner_iteration,
                                       batch_size=batch_size,
                                       verbose=self.verbose))
            # get L optimized
            self.L = self.AE.predict(self.L)

            # shrink S
            self.S = self.shrink_function(x=(x - self.L).reshape(x.size), eps=self.lambda_ / mu).reshape(x.shape)

            # criterion 1: Check if L and S are close enough to X
            c1 = linalg.norm(x - self.L - self.S, 'fro') / XFnorm
            # criterion 2: Check if L and S have converged since last interation
            c2 = np.min([mu, np.sqrt(mu)]) * linalg.norm(LS0 - self.L - self.S) / XFnorm

            self.STATS['c1'].append(c1)
            self.STATS['c2'].append(c2)

            if self.verbose:
                print(f"Errors: 1.{c1} 2.{c2}")

            if c1 < self.error and c2 < self.error:
                break

            # save L + S for c2 next iter
            LS0 = self.L + self.S

        if self.verbose:
            if i < self.max_iter - 1:
                print('Converged in %d steps' % i)
            else:
                print('Reached maximum iterations')


        return self.L, self.S

    def predict(self, x):
        return self.AE.predict(x)



