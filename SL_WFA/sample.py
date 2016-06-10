from load import Load

class Sample(object):

    def __init__(self, path, rows=0, columns=0):        
        l = Load(path).load_data(rows=rows, columns=columns)
        self.n_number = l[0]
        self.n_data = l[1]
        self.d_sample = l[2]
        self.d_prefix = l[3]
        self.d_suffix = l[4]
        self.d_factor = l[5]

    def n_number(self):
        return self.n_number

    def n_data(self):
        return self.n_data

    def d_sample(self):
        return self.d_sample

    def d_prefix(self):
        return self.d_prefix

    def d_suffix(self):
        return self.d_suffix

    def d_factor(self):
        return self.d_factor

    def select_rows(self, rows_bound=1000):
        l_rows = [] 
        temp = [([],self.d_prefix[()])]  
        idx = 0
        
        while temp and idx < rows_bound:
            lastWord = temp.pop()[0] 
            l_rows.append(tuple(lastWord))
            idx += 1
            for i in range(self.n_number):
                newWord = lastWord + [i]
                tnewWord = tuple(newWord) 
                if tnewWord in self.d_prefix:
                    temp.append((newWord, self.d_prefix[tnewWord]))
            temp = sorted(temp, key = lambda x: x[1])
        
        return l_rows


    def select_columns(self, columns_bound=1000):
        l_columns = [] 
        temp = [([],self.d_suffix[()])]  
        idx = 0
        
        while temp and idx < columns_bound:
            lastWord = temp.pop()[0] 
            l_columns.append(tuple(lastWord))
            idx += 1
            for i in range(self.n_number):
                newWord = lastWord + [i]
                tnewWord = tuple(newWord) 
                if tnewWord in self.d_suffix:
                    temp.append((newWord, self.d_suffix[tnewWord]))
            temp = sorted(temp, key = lambda x: x[1]) 
        
        return l_columns