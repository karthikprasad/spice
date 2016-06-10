from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np

train_file = 'data/1.spice.train'

def readfile(f):
	lengths = []
	sequences = []
	line = f.readline()
	l = line.split(" ")
	num_strings = int(l[0])
	alphabet_size = int(l[1])
	for n in range(num_strings):
		line = f.readline().rstrip()
		l = line.split(" ")
		seq = [[int(e)] for e in l[1:]]
		seq.append([-1])	# special stop symbol
		sequences = np.concatenate([seq]) if sequences == [] else np.concatenate([sequences, seq])
		lengths.append(int(l[0])+1)
	return alphabet_size, sequences, np.asarray(lengths)

alphabet, trainSequence, trainLengths = readfile(open(train_file,"r"))
print(trainSequence)

model = hmm.GaussianHMM(n_components=50).fit(trainSequence, trainLengths)
print(model.monitor_.converged)
joblib.dump(model, "learnedModels/hmm50_p1.pkl")

