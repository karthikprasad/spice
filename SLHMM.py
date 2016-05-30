#! /usr/bin/python
# @author: karthik
# @date: 20 May 2016

import cPickle
import numpy as np

class HMM(object):
    def __init__(self, num_hidden_states, num_observed_states,
                    transition_matrix=None, emission_matrix=None, ph0=None):
        '''
        @num_hidden_states: Number of hidden states in the HMM. (int)
        @num_observed_states: Number of observations in the HMM. (int)
        @transition_matrix: Matrix T. {T_ij = P(h_t+1 = i | h_t = j)} (np.ndarray)
                            1. shape = (num_hidden_states, num_hidden_states)
                            2. All the elements should be non-negative
        @emission_matrix: Matrix O. {O_ij = P(o_t = j | h_t = i)} (np.ndarray)
                          1. shape = (num_hidden_states, num_observed_states)
                          2. All the elements should be non-negative
        @ph0: {ph0_i = P(h_0 = i)} (np.ndarray)
                       1. shape = (num_hidden_states,); should be conveted to a column vector while operating
                       2. All the elements should be non-negative
        '''
        self._num_hidden_states   = num_hidden_states
        self._num_observed_states = num_observed_states

        # Build transition matrix, default is Identity matrix 
        if transition_matrix is None:
            self._transition_matrix = np.eye(num_hidden_states, dtype=np.float)
        else:
            # 1. check the shape of the matrix
            if not (transition_matrix.shape == (num_hidden_states, num_hidden_states)):
                raise ValueError("Transition matrix should be a %dx%d matrix" % (num_hidden_states, num_hidden_states))
            # 2. check the non-negativity of matrix
            if not np.all(transition_matrix >= 0):
                raise ValueError("Elements in transition matrix should be non-negative")
            # 3. normalize the matrix
            self._transition_matrix = transition_matrix
            normalizer = np.sum(transition_matrix, axis=1, keepdims=True)
            self._transition_matrix /= normalizer

        # Build emission matrix, default is Identity matrix
        if emission_matrix is None:
            self._emission_matrix = np.eye(num_hidden_states, num_observed_states)
        else:
            # 1. check the shape of the matrix
            if not (emission_matrix.shape == (num_hidden_states, num_observed_states)):
                raise ValueError("Emission matrix should be a %dx%d matrix" % (num_hidden_states, num_observed_states))
            # 2. check the non-negativity of matrix
            if not np.all(emission_matrix >= 0):
                raise ValueError("Elements in emission matrix should be non-negative")
            self._emission_matrix = emission_matrix
            normalizer = np.sum(emission_matrix, axis=1, keepdims=True)
            self._emission_matrix /= normalizer
            
        # Build first state distribution, default is uniform distribution
        if ph0 is None:
            self._ph0 = np.ones(num_hidden_states, dtype=np.float)
            self._ph0 /= num_hidden_states
        else:
            if not (ph0.shape[0] == num_hidden_states):
                raise ValueError("Initial distribution should have length: %d" % num_hidden_states)
            if not np.all(ph0 >= 0):
                raise ValueError("Elements in ph0ribution should be non-negative")
            self._ph0 = ph0
            self._ph0 /= np.sum(ph0)

        # First three order moments
        self._P_1 = np.zeros(self._num_observed_states, dtype=np.float)
        self._P_21 = np.zeros((self._num_observed_states, self._num_observed_states), dtype=np.float)
        self._P_3x1 = np.zeros((self._num_observed_states, self._num_observed_states, self._num_observed_states), dtype=np.float)
        
    ##############################################################################
    # Getters
    ##############################################################################        
    @property
    def ph0(self):
        return self._ph0
    
    @property
    def transition_matrix(self):
        return self._transition_matrix
    
    @property
    def emission_matrix(self):
        return self._emission_matrix

    ##############################################################################
    # Actions
    ##############################################################################

    def fit(self, sequences, rank_hyperparameter=None, verbose=False):
        '''
        Solve the learning problem with HMM.
        @sequences: List of observed sequences (each of possible variable length) (list of np.array)
        '''
        # Set default value of rank-hyperparameter
        rank_hyperparameter = self._num_hidden_states if rank_hyperparameter is None else rank_hyperparameter
        # Generate list of triples to train on
        triple_list = np.array([sequence[idx: idx+3] for sequence in sequences
                           for idx in xrange(len(sequence)-2)], dtype=np.int)

        # Parameter estimation
        # Frequency based estimation
        for sq in triple_list:
            self._P_1[sq[0]] += 1
            self._P_21[sq[1], sq[0]] += 1
            self._P_3x1[sq[1], sq[2], sq[0]] += 1
        # Normalization of P_1, P_21, P_3x1
        norm = np.sum(self._P_1)
        self._P_1 /= norm
        # Normalize the joint distribution of P_21        
        norm = np.sum(self._P_21)
        self._P_21 /= norm
        # Normalize the joint distribution of P_3x1
        norm = np.sum(self._P_3x1)
        self._P_3x1 /= norm

        # Singular Value Decomposition
        # Keep all the positive singular values
        (U, _, V) = np.linalg.svd(self._P_21)
        U = U[:, 0:rank_hyperparameter]
        V = V[0:rank_hyperparameter, :]
        # Compute b1, binf and Bx
        # self.factor = (P_21^{T} * U)^{+}, which is used to accelerate the computation
        p21tui = np.linalg.pinv(np.dot(self._P_21.T, U))
        self._b_1 = np.dot(U.T, self._P_1)        
        self._b_inf = np.dot(p21tui, self._P_1)
        self._B_x = np.zeros((self._num_observed_states, rank_hyperparameter, rank_hyperparameter), dtype=np.float)        
        for x in xrange(self._num_observed_states):
            self._B_x[x] = np.dot(np.dot(U.T, self._P_3x1[x]), p21tui.T)

    def predict(self, sequence_prefix):
        '''
        Predict the next element of the sequence given the prefix by running the forward algorithm
        @sequence_prefix: Observed sequence prefix (np.array)
        '''
        
        # classical EM prediction
        '''
        return np.sum(self._forward(sequence_prefix)[-1, :])  # check later
        '''

        # finding the total joint probability
        '''
        prob = self._b_1
        for ob in sequence_prefix:
            prob = np.dot(self._B_x[ob], prob)
        prob = np.dot(self._b_inf.T, prob)
        return prob
        '''

        # SL Prediction
        # compute b_t
        # b_t+1 = (B_x[xt] . b_t) / (b_inf.T . B_x[xt] . b_t)
        b_t = self._b_1
        for xt in sequence_prefix:
            numerator = np.dot(self._B_x[xt],b_t)
            b_t = numerator / np.dot(self._b_inf.T, numerator)
        # calculate conditional prob
        conditional_prob = {}
        denominator = 0.0
        for x in xrange(self._num_observed_states):
            denominator += np.dot(self._b_inf.T, np.dot(self._B_x[x], b_t))
        for next_char in xrange(self._num_observed_states):
            numerator = np.dot(self._b_inf.T, np.dot(self._B_x[next_char], b_t))
            conditional_prob[next_char] = numerator / denominator
        return conditional_prob

    def predict_joint(self, sequence_prefix):
        prob = self._b_1
        for ob in sequence_prefix:
            prob = np.dot(self._B_x[ob], prob)
        ret = {}
        for x in xrange(self._num_observed_states):
            ret[x] = np.dot(self._b_inf.T, np.dot(self._B_x[x], prob))
        return ret

        
     
    ###############################################################
    # Algorithims (from assignment 1 just in case)
    ###############################################################
    def _forward(self, sequence):
        '''
        @sequence: Observed sequence (np.array)
        @desc: Computing the forward-probability: P(o_1,o_2,...,o_t, h_t=i)
        '''
        L = len(sequence)
        f = np.zeros((L, self._num_hidden_states), dtype=np.float)
        f[0, :] = self._emission_matrix[:, sequence[0]] * self._ph0
        for t in xrange(1, L):
            f[t, :] = np.multiply(self._emission_matrix[:, sequence[t]], np.dot(f[t-1, :], self._transition_matrix))
        return f
    
    def _backward(self, sequence):
        '''
        @sequence: Observed sequence (np.array)
        @desc: Computing the backward-probability: P(o_t+1, ..., o_L | h_t)
        '''
        L = len(sequence)
        r = np.zeros((L, self._num_hidden_states), dtype=np.float)
        r[L-1, :] = 1.0
        for t in xrange(L-1, 0, -1):
            r[t-1, :] = np.multiply(self._emission_matrix[:, sequence[t]], r[t, :].T)
            r[t-1, :] = np.dot(self._transition_matrix, r[t-1, :])
        return r
    
    ######################################################
    # Model persistence
    ######################################################
    @staticmethod
    def dump_model(filename, hmm):
        with file(filename, "wb") as fout:
            cPickle.dump(hmm, fout)
    
    @staticmethod
    def load_model(filename):
        with file(filename, "rb") as fin:
            model = cPickle.load(fin)
            return model

def load_training_data(training_file):
    lines = []
    with open(training_file, 'r') as f:
        lines = f.readlines()
    num_observed_states = int(lines[0].strip().split()[-1]) + 1
    lines = lines[1:]
    sequences = []
    for line in lines:
        sequence = map(np.float, (line.strip()+' -1').split()[1:])
        sequences.append(np.asarray(sequence))
    return sequences, num_observed_states



if __name__ == '__main__':
    training_file = 'codeBlue/data/0.spice.train'
    sequences, num_obs_states = load_training_data(training_file)
    model = HMM(num_obs_states, num_obs_states)
    model.fit(sequences)


