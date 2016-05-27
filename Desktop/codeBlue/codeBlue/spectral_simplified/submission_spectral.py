problem = '2'
user_id = '81'

train_file = problem + ".spice.train"
test_file = problem + ".spice.public.test"

rank = 16
rows = 7
columns = 7

name = "rank_" + str(rank) + "_sparse_rows_columns_" + str(rows)

def learn(train_file, rank):
    import learning as LC
    from sample import Sample

    sample_data = Sample(path=train_file, rows=rows, columns=columns)
        
    sample_automaton = LC.Learning(sample_instance=sample_data)
                
    A = sample_automaton.learnAutomaton(rank=rank, rows=rows, columns=columns)

    Ap = A.transform()

    return Ap

def next_symbols_ranking(model, prefix, k=5):
    word = prefix.split()
    word = [int(i) for i in word][1:]

    # Compute the weight of the prefix
    p_w = model.val(word)
    for i in range(model.n_number):
        p_w -= model.val(word+[i])

    # Symbol -1 correspond to end of sequence
    # If the weight is negative it does not carry any semantic
    l = [(-1, max(p_w, 0))]
    s = max(p_w, 0)

    # Compute the weight of the prefix concatenated to each possible symbol
    for i in range(model.n_number):
        l.append((i, max(model.val(word+[i]), 0)))
        s += max(model.val(word+[i]), 0)

    # Sort the symbol by decreasing weight
    l = sorted(l, key=lambda x: -x[1])

    if s != 0:
        # At least one symbol has a strictly positive weight
        # Return a string containing the sorted k most probable next symbols separted by spaces
        mot = trans_string([x[0] for x in l][0:k])
        return mot
    else:
        # All symbols have a non-positive weight in the model
        # Return the k first symbols...
        return trans_string([x for x in range(-1, k-1)])

def trans_string(list):
    """ Transform a list of interger into a string of elements separated by a space """
    mot = ""
    for w in list:
        mot +=  str(w) + ' '
    return mot

def get_first_prefix(test_file):
    """ get the only prefix in test_file """
    f = open(test_file)
    prefix = f.readline()
    f.close()
    return prefix

def formatString(string_in):
    """ Replace white spaces by %20 """
    return string_in.strip().replace(" ", "%20")

# learn the model
print ("Start learning")
model = learn(train_file, rank)
print ("Finish learning")

# get the test first prefix: the only element of the test set
first_prefix = get_first_prefix(test_file)

# get the next symbol ranking on the first prefix
ranking = next_symbols_ranking(model, first_prefix)

print ("Prefix number: 1 Ranking: " + ranking + " Prefix: " + first_prefix)

# transform ranking to follow submission format (with %20 between symbols)
ranking = formatString(ranking)

# transform the first prefix to follow submission format
first_prefix = formatString(first_prefix)

# create the url to submit the ranking
url_base = 'http://spice.lif.univ-mrs.fr/submit.php?user=' + user_id +\
           '&problem=' + problem + '&submission=' + name + '&'
url = url_base + 'prefix=' + first_prefix + '&prefix_number=1' + '&ranking=' + ranking

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
    ranking = next_symbols_ranking(model, prefix)
    
    if prefix_number % 2 == 0:
        print("Prefix number: " + str(prefix_number) + " Ranking: " + ranking + " Prefix: " + prefix)
    
    # Format the ranking
    ranking = formatString(ranking)

    # create prefix with submission needed format
    prefix=formatString(prefix)

    # Create the url with your ranking to get the next prefix
    url = url_base + 'prefix=' + prefix + '&prefix_number=' + str(prefix_number) + '&ranking=' + ranking

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
