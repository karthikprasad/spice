from __future__ import division, print_function
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as lin
import math
from sample import Sample
from hankel import Hankel
import automaton as AC

class Learning(object):

    def __init__(self, sample_instance):
        self.sample_object = sample_instance

    def sample_object(self):
        return self.sample_object

    @staticmethod
    def buildAutomaton(hankels, nbL, rank):
        print ("Start building Automaton from Hankel matrix")
        
        hankel = hankels[0]
        [u, s, v] = lin.svds(hankel, k=rank)
        ds = np.diag(s)
        pis = np.linalg.pinv(v)
        del v
        pip = np.linalg.pinv(np.dot(u, ds))
        del u, ds
        init = hankel[0, :].dot(pis)[0, :]
        term = np.dot(pip, hankel[:, 0].toarray())[:, 0]
        trans = []
        for x in range(nbL):
            hankel = hankels[x+1]
            trans.append(np.dot(pip, hankel.dot(pis)))        
        A = AC.Automaton(nbL, rank, init, term, trans)

        print ("Finish building Automaton")
        return A

    def learnAutomaton(self, rank, rows=0, columns=0):
        hankels = Hankel(sample_instance=self.sample_object, rows=rows, columns=columns).hankels
        matrix_shape =min(hankels[0].shape)
        if (min(hankels[0].shape) < rank) :
            raise ValueError("Rank "+str(rank)+" should <= "+"Hankel matrix shape "+str(matrix_shape))
        A = self.buildAutomaton(hankels=hankels, nbL=self.sample_object.n_number, rank=rank)
        A.initial = A.initial / self.sample_object.n_data
        
        return A
    