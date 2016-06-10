from __future__ import division, print_function
import scipy.sparse as sps
import numpy as np

class Hankel(object):

    def __init__(self, sample_instance, rows=0, columns=0):
        self.n_number = sample_instance.n_number
        self.n_data = sample_instance.n_data
        self.hankels = self.build(d_sample=sample_instance.d_sample,
                                  d_prefix=sample_instance.d_prefix,
                                  d_suffix=sample_instance.d_suffix,
                                  d_factor=sample_instance.d_factor,
                                  rows=rows, columns=columns)

    def n_number(self):
        return self.n_number

    def n_data(self):
        return self.n_data

    def build(self, d_sample, d_prefix, d_suffix, d_factor, rows, columns):
        print ("Start computing Hankel matrix")

        dict_first=d_prefix
        dict_second=d_suffix
        
        l_rows = [w for w in dict_first if len(w) <= rows]
        
        l_columns = [w for w in dict_second if len(w) <= columns]

        (sorted_rows, sorted_columns) = self.sorting(l_rows, l_columns)

        n_rows = len(l_rows)
        n_columns = len(l_columns)
        s_rows = set(l_rows)
        s_columns = set(l_columns)

        hankels = [sps.dok_matrix((n_rows, n_columns)) for i in range(self.n_number+1)]
        
        for w in d_sample:
            for i in range(len(w)+1):
                if w[:i] in s_rows:
                    if w[i:] in s_columns:
                        hankels[0][sorted_rows[w[:i]], sorted_columns[w[i:]]] = d_sample[w]
                    if (i < len(w) and w[i+1:] in s_columns):
                        hankels[w[i]+1][sorted_rows[w[:i]],sorted_columns[w[i+1:]]] = d_sample[w]
        
        print ("Finish computing Hankel matrix")
        return hankels

    def sorting(self, l_rows, l_columns):
        n_rows = len(l_rows)
        l_rows = sorted(l_rows, key=lambda x: (len(x), x))
        sorted_rows = {l_rows[i]: i for i in range(n_rows)}
        n_columns = len(l_columns)
        l_columns = sorted(l_columns, key=lambda x: (len(x), x))
        sorted_columns = {l_columns[i]: i for i in range(n_columns)}

        return (sorted_rows, sorted_columns)