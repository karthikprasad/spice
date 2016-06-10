class Load(object):

    def __init__(self, path):
        self.path = path

    def load_data(self, rows=0, columns=0):
        d_sample, d_prefix, d_suffix, d_factor = {}, {}, {}, {}
        f = open(self.path, "r")
        line = f.readline()
        l = line.split()
        n_data, n_number = int(l[0]), int(l[1])
        line = f.readline()

        while line:
            l = line.split()
            w = () if int(l[0]) == 0 else tuple([int(x) for x in l[1:]])               
            d_sample[w] = d_sample[w] + 1 if w in d_sample else 1            
            d_prefix[()] = d_prefix[()] + 1 if () in d_prefix else 1            
            d_suffix[()] = d_suffix[()] + 1 if () in d_suffix else 1            
            for i in range(len(w)):
                if i < rows:
                    d_prefix[w[:i + 1]] = d_prefix[w[:i + 1]] + 1 if w[:i + 1] in d_prefix else 1
                if i < columns:
                    d_suffix[w[-(i + 1):]] = d_suffix[w[-(i + 1):]] + 1 if w[-(i + 1):] in d_suffix else 1                
            line = f.readline()
        
        f.close()
        return n_number, n_data, d_sample, d_prefix, d_suffix, d_factor