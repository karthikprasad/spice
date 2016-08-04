#! /usr/bin/python
# @date: 20 May 2016
# @date: 04 Jun 2016

import cPickle
import numpy as np
import urllib2 as ul

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

    ###############################################################
    # Algorithims (from assignment 1)
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

    ##############################################################################
    # Actions
    ##############################################################################

    def fit(self, sequences, verbose=False):
        '''
        Solve the learning problem with HMM.
        @sequences: List of observed sequences (each of possible variable length) (list of np.array)
        '''
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

        # Set rank
        rank_hyperparameter = np.linalg.matrix_rank(self._P_21)

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

    def predict_BW(self, sequence_prefix):
        return np.sum(self._forward(sequence_prefix)[-1,:])

    def predict(self, sequence_prefix):
        '''
        Spectral Learning of HMM Prediction
        Predict the next element of the sequence given the prefix by running the forward algorithm
        @sequence_prefix: Observed sequence prefix (np.array)
        '''

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


def order_probabilities(probability_dict):
    '''
    order the symbols based on the probabilities supplied as a dictionary in decreaing order
    @probability_dict: dict.
    '''
    tuple_list = [(k,v) for k,v in probability_dict.iteritems()]
    tuple_list.sort(key=lambda x: x[1], reverse=True)  # sort on the probability value
    ordered_seq = [k for k,v in tuple_list]
    return ordered_seq

def get_ranking(probabilities):
    '''
    obtain the ranked order of symbols given a the next state probability distribution
    '''
    ranked_order = order_probabilities(probabilities)[:5]
    #replace the end of state sequence with -1
    eos_state = max(ranked_order)  
    ranked_order = [str(x) if x != eos_state else str(-1) for x in ranked_order]
    ranking_str = '%20'.join(ranked_order)
    return ranking_str


def load_training_data(training_file):
    '''
    load training data given file name
    '''
    lines = []
    with open(training_file, 'r') as f:
        lines = f.readlines()
    num_observed_states = int(lines[0].strip().split()[-1]) + 1
    eos_state = str(num_observed_states - 1)  # the end of a sequence is represented by another state
    lines = lines[1:]
    sequences = []
    for line in lines:
        sequence = map(np.int, (line.strip() + ' ' + eos_state).split()[1:])
        sequences.append(np.asarray(sequence))
    return sequences, num_observed_states


def make_submission(problem_num=0):
    '''
    repeatedly make submissions to the spice website for a particular problem
    '''
    problem_num = str(problem_num)
    user_id = '81'
    name = 'codeBlue_app2'
    url_base = 'http://spice.lif.univ-mrs.fr/submit.php?user=' + user_id + \
                '&problem=' + problem_num + '&submission=' + name + '&'

    training_file = 'data/' + problem_num + '.spice.train'
    testing_file  = 'data/' + problem_num + '.spice.public.test'

    # train
    sequences, num_obs_states = load_training_data(training_file)  # end of sequence state has the maximum number
    model = HMM(num_obs_states, num_obs_states)
    model.fit(sequences)

    def get_prefix_from_file(prefix_file):
        with open(prefix_file, 'r') as f:
            line = f.readline()
        line = line.strip().split()
        return line

    def get_prefix_sequence(prefix):
        return map(np.int, prefix[1:])

    def get_submission_url(prefix_num, prefix):
        prefix_str = '%20'.join(prefix)
        prefix_sequence = get_prefix_sequence(prefix)
        probabilities = model.predict(prefix_sequence)
        ranking_str = get_ranking(probabilities)
        print 'Prefix number: ' + str(prefix_num) + ' Prefix: ' + ' '.join(prefix) + ' Ranking: ' + ranking_str.replace('%20', ' ')
        # Create the url with your ranking to get the next prefix
        url = url_base + 'prefix=' + prefix_str + '&prefix_number=' +\
            str(prefix_num) + '&ranking=' + ranking_str
        return url

    # test on first prefix
    prefix_num = 1
    prefix = get_prefix_from_file(testing_file)
    url = get_submission_url(prefix_num, prefix)
    # initiate submission
    response = ul.urlopen(url)
    content = response.read()
    list_element = content.split()
    head = str(list_element[0])
    
    while head != '[Error]' and head != '[Success]':
        prefix_num += 1
        prefix = content[:-1].strip().split()
        url = get_submission_url(prefix_num, prefix)
        # initiate submission
        response = ul.urlopen(url)
        content = response.read()
        list_element = content.split()
        head = str(list_element[0])

    # Post-treatment
    # The score is the last element of content (in case of a public test set)
    print content
    score = list_element[-1]
    print score


if __name__ == '__main__':
    #make_submission(0)
    for i in range(0,5):
      make_submission(i)


