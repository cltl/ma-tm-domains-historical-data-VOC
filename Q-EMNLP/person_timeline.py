# Extracts the top n most frequently mentioned persons over time, normalized by the number of missives per decade.

import pandas as pd
import numpy as np
import json
import glob
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

###########################################################
#                      USER INPUTS                        #
###########################################################

# dictionary with document dates
gm_dates = 'gm_dates.json'

# folder with all the system labeled files
system_output_folder = 'system_output'

# top n most frequently mentioned persons
top_n = 5

# Clustering y/n ?
to_cluster_or_not_to_cluster = 'n'

###########################################################
#                        FUNCTIONS                        #
###########################################################

def preprocess(file):
    '''
    Return:
    {2: [['17', 'B-DATE'], ['december', 'I-DATE'], ['1617', 'I-DATE']],
     9: [['Straete', 'B-LOC'], ['Sunda', 'B-LOC']],
     11:[['Jan', 'B-PER'], ['Pietersz', 'I-PER']]}
    '''
    data = pd.read_csv(file, sep='\t', names=['sent_id','token','tag'])
    data = data[data.tag != 'O'] # remove rows with 'O' tags
    data = data.groupby('sent_id')[['token', 'tag']].apply(lambda g: g.values.tolist()).to_dict()
    return data

def extract_entities(file):
    data = preprocess(file)

    entities = {}
    for sent_id, rows in data.items():
        sent_ents = []
        for row in rows:
            token = row[0]
            IOB_tag = row[1][0]
            ent_type = row[1].replace('B-','').replace('I-','')

            if IOB_tag == 'B':
                sent_ents += [[ent_type, [token]]]

            elif IOB_tag == 'I':
                try:
                    sent_ents[-1][-1] += [token]
                except:
                    continue

        sent_ents = [(r[0], ' '.join(r[1])) for r in sent_ents ]
        entities[sent_id] = sent_ents

    return entities

def extract_persons(file):
    all_entities = extract_entities(file)

    persons = []
    for sent in all_entities.values():
        for ent_type, ent in sent:
            if ent_type == 'PER':
                persons.append(ent.lower())

    return persons

def get_decade(file):
    filename = file.split('/')[-1].split('.')[0].replace('system-','')
    date = doc_dates[filename]
    if date != None:
        year = re.findall(r'\d{4}', date)
        if year != []:
            year = year[-1]
        else:
            return 'NO YEAR'
    else:
        return 'NO YEAR'

    decade = int(year) // 10 * 10
    return decade

def damerau_levenshtein(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return 1.0 - d[lenstr1-1,lenstr2-1] / max(lenstr1, lenstr2)

def clustering(lst):
    column_set = list(set(lst))

    # Similarity matrix
    matrix = [[damerau_levenshtein(i,j) for i in column_set] for j in tqdm(column_set)]
    clusters = DBSCAN(eps=0.6, min_samples=1).fit_predict(matrix)

    # Give each item a cluster ID
    cluster_dict = {k:v for k,v in zip(column_set, clusters)}
    column_clusters = [cluster_dict[i] for i in lst]
    dfx = pd.DataFrame(zip(lst, column_clusters), columns=['X','cluster'])

    # Replace cluster ID with the most common item in the cluster
    agg_func = lambda s: Counter(s['X']).most_common()[0][0]
    cluster_conversions = dfx.groupby('cluster').apply(agg_func)
    cluster_conversions = {k:v for k,v in cluster_conversions.items()}
    dfx = dfx.replace({'cluster' : cluster_conversions})

    return dfx['cluster']


###########################################################
#          EXTRACTING PERSONS FROM SYSTEM OUTPUT          #
###########################################################

extract_persons(files[1])

# :: Dictionary with the date for each document ::
with open(gm_dates) as json_file:
    doc_dates = json.load(json_file)

# :: List of system output files ::
files = glob.glob(system_output_folder + '/*.txt')

# :: Extract persons and decades ::
persons_and_decade = []
for file in tqdm(files):
    decade = get_decade(file)
    if decade != 'NO YEAR':
        persons = extract_persons(file)
        persons_and_decade += [(p, decade) for p in persons]

df = pd.DataFrame(persons_and_decade, columns=['person','decade'])
print(f':: No of persons extracted: {len(df)} ::')

###########################################################
#                       CLUSTERING                        #
###########################################################

if to_cluster_or_not_to_cluster = 'y':
    df['person'] = clustering(df['person'])

###########################################################
#              FILTERING & NORMALIZATION                  #
###########################################################

# Count no. of missives per decade
doc_dates = {k:v for k,v in doc_dates.items() if v != None}

gm_decades = []
for date in doc_dates.values():
    year = re.findall(r'\d{4}', date)
    if year != []:
        year = year[-1]
        decade = int(year) // 10 * 10
        gm_decades.append(decade)

missives_per_decade = Counter(gm_decades)

# :: Filter out all but the top n most frequently mentioned persons
exclude = ['sultan','radja','may.1','conink','conincq']
prominent_persons = Counter(df['person']).most_common(topn + 10)
prominent_persons = [(k,v) for k,v in prominent_persons if k not in exclude][:top_n]
prominent_persons = {k:v for k,v in prominent_persons if k not in exclude}

df = df[df['person'].isin(prominent_persons)]

# Count absolute mentions per decade
counts = df.groupby(['person','decade']).size()
counts = counts.to_dict()

# Normalize counts by dividing by no. of missiven per decade
normalized_counts = {}
for k, count in counts.items():
    person = k[0]
    decade = k[1]
    normalized = count / missives_per_decade[decade]
    # print(count, normalized)

    if person not in normalized_counts:
        normalized_counts[person] = {k:0 for k in sorted(missives_per_decade.keys())}
        normalized_counts[person][decade] = normalized
    else:
        normalized_counts[person][decade] = normalized


###########################################################
#                      VISUALIZATION                      #
###########################################################

# Plot timeline
plt.style.use("ggplot")
plt.figure(figsize=(10,4))
for name, values in normalized_counts.items():
    tuples = values.items()
    x = [i[0] for i in tuples]
    y = [i[1] for i in tuples]
    plt.plot(x, y, marker = '.', linewidth=3)
plt.xlabel('year', fontsize=13)
plt.ylabel('avg. mentions per missive', fontsize=13)
# plt.ticklabel_format(axis="y", style="plain")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(normalized_counts.keys(), loc='best')
plt.show()
