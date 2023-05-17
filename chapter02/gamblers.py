import numpy as np
from numpy.random import choice
from scipy.special import softmax


class Gambler:
    """ Base class for repeatable pseudo-random Bandit algorithms """
    def __init__(self):
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def arm(self):
        raise NotImplementedError()

    def update(self, arm, reward):
        pass

    def reset(self):
        pass


class AveragingGambler(Gambler):
    """ Simple Bandit algorithm with true reward averaging per Chapter 2.4 """
    def __init__(self, n=10, epsilon=0.1):
        super().__init__()
        self.n, self.epsilon = n, epsilon
        self.N = None
        self.Q = None
        self.reset()

    def arm(self):
        if self.np_random.rand() < self.epsilon:
            arm = self.np_random.randint(0, self.n)
        else:
            arm = np.argmax(self.Q)
        return arm

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

    def reset(self):
        self.N = np.zeros([self.n])
        self.Q = self.np_random.rand(self.n) * 1e-5


class EpsilonGreedyGambler(Gambler):
    """ Simple Bandit algorithm with fixed learning step per Chapter 2.5 """
    def __init__(self, n=10, epsilon=0.1, alpha=0.1):
        super().__init__()
        self.n, self.epsilon, self.alpha = n, epsilon, alpha
        self.Q = None
        self.reset()

    def arm(self):
        if self.np_random.rand() < self.epsilon:
            arm = self.np_random.randint(0, self.n)
        else:
            arm = np.argmax(self.Q)
        return arm

    def update(self, arm, reward):
        self.Q[arm] += (reward - self.Q[arm]) * self.alpha

    def reset(self):
        self.Q = self.np_random.rand(self.n) * 1e-5


class ReinforcementComparisonGambler(Gambler):
    """ Simple Bandit algorithm per Chapter 2.8 """
    def __init__(self, n=10, beta=0.1, alpha=0.1):
        super().__init__()
        self.n, self.beta, self.alpha = n, beta, alpha
        self.p = None
        self.r = 0
        self.reset()

    def arm(self):
        probs = softmax(self.p)
        arm = self.np_random.choice(np.arange(self.n), 1, p=probs)
        return arm[0]

    def update(self, arm, reward):
        self.p[arm] += (reward - self.r) * self.beta
        self.r += (reward - self.r) * self.alpha

    def reset(self):
        self.p = np.zeros(self.n)
        self.r = 0


class PursuitGambler(EpsilonGreedyGambler):
    """ Simple Bandit algorithm per Chapter 2.9 """

    def __init__(self, n=10, epsilon=0.1, alpha=0.1, beta=0.1):
        self.beta = beta
        self.p = None
        super().__init__(n, epsilon, alpha)

    def arm(self):
        arm = self.np_random.choice(np.arange(self.n), 1, p=self.p)
        return arm[0]

    def update(self, arm, reward):
        a_star = np.argmax(self.Q)

        for i in range(self.n):
            if i == a_star:
                self.p[i] += (1 - self.p[i]) * self.beta
            else:
                self.p[i] += (0 - self.p[i]) * self.beta

        self.Q[arm] += (reward - self.Q[arm]) * self.alpha

    def reset(self):
        self.Q = self.np_random.rand(self.n) * 1e-5
        self.p = np.ones(self.n) / self.n

class OptimisticGambler(EpsilonGreedyGambler):
    """ Optimistic Initial Values algorithm per Chapter 2.6 """
    def __init__(self, n=10, epsilon=0.0, alpha=0.1, init=10.0):
        self.init = init
        super().__init__(n, epsilon, alpha)

    def reset(self):
        self.Q = np.ones([self.n]) * self.init


class UCBGambler(Gambler):
    """ Upper Confidence Bound Action Selection algorithm per Chapter 2.7 """
    def __init__(self, n=10, c=1.0):
        super().__init__()
        self.n, self.c = n, c
        self.N = None
        self.Q = None
        self.reset()

    def ucb(self):
        t = np.sum(self.N) + 1
        ucb = self.Q + self.c * np.sqrt(np.log(t) / (self.N + 1e-5))
        return ucb

    def arm(self):
        ucb = self.ucb()
        arm = np.argmax(ucb)
        return arm

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

    def reset(self):
        self.N = np.zeros([self.n])
        self.Q = self.np_random.rand(self.n) * 1e-5


class GradientGambler(Gambler):
    """ Gradient Bandit algorithm per Chapter 2.8 """
    def __init__(self, n=10, alpha=0.1):
        super().__init__()
        self.n, self.alpha = n, alpha
        self.last_p = None
        self.H = None
        self.R = None
        self.reset()

    def p(self):
        exps = np.exp(self.H - np.max(self.H))
        p = exps / exps.sum()
        self.last_p = p
        return p

    def arm(self):
        arms = [arm for arm in range(self.n)]
        arm = self.np_random.choice(arms, p=self.p())
        return arm

    def update(self, arm, reward):
        others = [a for a in range(self.n) if a != arm]
        p = self.last_p

        self.H[arm] += self.alpha * (reward - self.R) * (1 - p[arm])
        self.H[others] -= self.alpha * (reward - self.R) * p[others]

        self.R += self.alpha * (reward - self.R)

    def reset(self):
        self.last_p = None
        self.H = self.np_random.rand(self.n) * 1e-2
        self.R = 0.0
