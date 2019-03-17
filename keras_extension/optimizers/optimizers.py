"""Customized optimizer classes.
"""
from keras import backend as K
from keras.optimizers import Optimizer
from keras.legacy import interfaces


class AdaBound(Optimizer):
    """AdaBound optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        terminal_bound: float >=0. Final(Convergent) Learning rate.
        lower_bound: float >=0. Initial lower bound of Learning rate.
        upper_bound: float >=0. Initial upper bound of Learning rate.
    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
        - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](
           https://arxiv.org/abs/1902.09843v1)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0.,
                 terminal_bound=0.1, lower_bound=0., upper_bound=None, **kwargs):
        super(AdaBound, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            if upper_bound is None:
                upper_bound = terminal_bound * 2.
            self.terminal_bound = K.variable(terminal_bound, name='terminal_bound')
            self.lower_bound = K.variable(lower_bound, name='lower_bound')
            self.upper_bound = K.variable(upper_bound, name='upper_bound')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            eta_l_t = self.terminal_bound - \
                (self.terminal_bound - self.lower_bound) / \
                K.pow((1. - self.beta_2), t+1)
            eta_u_t = self.terminal_bound + \
                (self.upper_bound - self.terminal_bound) / \
                K.pow((1. - self.beta_2), t)

            clipped_lr_t = K.minimum(
                K.maximum(lr_t / (K.sqrt(v_t) + self.epsilon), eta_l_t), eta_u_t)
            p_t = p - clipped_lr_t * m_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'terminal_bound': float(K.get_value(self.terminal_bound)),
                  'upper_bound': float(K.get_value(self.upper_bound)),
                  'lower_bound': float(K.get_value(self.lower_bound)),
                  'epsilon': self.epsilon}
        base_config = super(AdaBound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
