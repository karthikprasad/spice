# State the problem number
problem_number = '1'

# and the user id (given during registration)
user_id = '81'

# name of this submission (no space or special character)

name = "hxxmx100"

train_file = 'data/1.spice.train'
prefix_file = 'data/1.spice.public.test'


import numpy as np 
from sys import *
from sklearn.externals import joblib

model = joblib.load("learnedModels/hmm100_p1.pkl")

def hmmRank(prefix, alphabet):
	score=[]
	seq = [[int(e)] for e in prefix]
	for s in alphabet:	
		seq.append([s])
		score.append((s,model.score_samples(np.concatenate([seq]))[0]))
		seq.pop()
	score=sorted(score, key=lambda x: -x[1])
	return [x[0] for x in score]

def get_first_prefix(test_file):
    """ get the only prefix in test_file """
    f = open(test_file)
    prefix = f.readline()
    f.close()
    return prefix

def list_to_string(l):
    s=str(l[0])
    for x in l[1:]:
        s+= " " + str(x)
    return(s)

def formatString(string_in):
    """ Replace white spaces by %20 """
    return string_in.strip().replace(" ", "%20")

print('Set of symbols')
num_syms = int(get_first_prefix(train_file).rstrip().split()[1])
alphabet = [ i for i in range (num_syms)]
alphabet.append(-1)
print(alphabet)

# get the test first prefix: the only element of the test set
first_prefix = get_first_prefix(prefix_file)
prefix_number=1

# get the next symbol ranking on the first prefix
p=first_prefix.split()
prefix=[int(i) for i in p[1:len(p)]]
ranking = hmmRank(prefix, alphabet)
ranking_string=list_to_string(ranking[:5])

print("Prefix number: " + str(prefix_number) + " Ranking: " + ranking_string + " Prefix: " + first_prefix)

# transform the first prefix to follow submission format
first_prefix = formatString(first_prefix)

# transform the ranking to follow submission format
ranking_string=formatString(ranking_string)

# create the url to submit the ranking
url_base = 'http://spice.lif.univ-mrs.fr/submit.php?user=' + user_id +\
    '&problem=' + problem_number + '&submission=' + name + '&'
url = url_base + 'prefix=' + first_prefix + '&prefix_number=1' + '&ranking=' +\
    ranking_string

# Get the website answer for the first prefix with this ranking using this
# submission name
try:
    # Python 2.7
    import urllib2 as ur
    orl2 = True
except:
    #Python 3.4
    import urllib.request as ur
    orl2 = False

response = ur.urlopen(url)
content = response.read()

if not orl2:
    # Needed for python 3.4...
    content= content.decode('utf-8')

list_element = content.split()
head = str(list_element[0])

prefix_number = 2

while(head != '[Error]' and head != '[Success]'):
    prefix = content[:-1]
    # Get the ranking
    p=prefix.split()
    prefix_list= list()
    prefix_list=[int(i) for i in p[1:len(p)]]
    ranking = hmmRank(prefix_list, alphabet)
    ranking_string=list_to_string(ranking[:5])
    
    print("Prefix number: " + str(prefix_number) + " Ranking: " + ranking_string + " Prefix: " + prefix)
    
    # Format the ranking
    ranking_string = formatString(ranking_string)
    
    # create prefix with submission needed format
    prefix=formatString(prefix)
    
    # Create the url with your ranking to get the next prefix
    url = url_base + 'prefix=' + prefix + '&prefix_number=' +\
        str(prefix_number) + '&ranking=' + ranking_string
    
    # Get the answer of the submission on current prefix
    response = ur.urlopen(url)
    content = response.read()
    if not orl2:
        # Needed for Python 3.4...
        content= content.decode('utf-8')
    
    list_element = content.split()
    # modify head in case it is finished or an erro occured
    head = str(list_element[0])
    # change prefix number
    prefix_number += 1

# Post-treatment
# The score is the last element of content (in case of a public test set)
print(content)

list_element = content.split()
score = (list_element[-1])
print(score)

