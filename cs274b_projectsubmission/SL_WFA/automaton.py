import numpy as np

class Automaton(object):

    def __init__(self, n_number=0, rank=0, initial=[], final=[], transitional=[]):
        self.n_number = n_number
        self.rank = rank
        self.initial = initial
        self.final = final
        self.transitional = transitional
        self.checkedConv = False
        self.absConv = False

    def final(self):
        return self.final

    def initial(self):
        return self.initial

    def transitional(self):
        return self.transitional

    def rank(self):
        return self.rank

    def n_number(self):
        return self.n_number

    def absConv(self):
        if not self.checkedConv:
            self.calcAbsConv()
            self.checkedConv = True
        return self.absConv

    def transform(self):
        A = Automaton(self.n_number, self.rank, self.initial, self.final, self.transitional)
        m_sigma = np.zeros(self.rank)
        for m in self.transitional:
            m_sigma = m_sigma + m
        m = np.eye(self.rank) - m_sigma
        im = np.linalg.inv(m)        
        A.final = np.dot(im, A.final)

        return A

    def val(self, word):
        u = self.initial
        final = self.final
        for x in word:
            u = np.dot(u, self.transitional[x])
        return np.dot(u, final)

    def calcAbsConv(self):
        m = np.zeros([self.rank, self.rank])
        for x in range(self.n_number):
            m = m + abs(self.transitional[x])
        if max(abs(np.linalg.eigvals(m))) < 1:
            self.absConv = True
        else:
            self.absConv = False
