# Extracts the top n most frequently traded goods and quantities from texts, aggregates the quantities per decade, which is visualized on a timeline.

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

###########################################################
#                      USER INPUTS                        #
###########################################################

# dictionary with document dates
gm_dates = 'gm_dates.json'

# folder with all the system labeled files
system_output_folder = 'system_output'

# dictionary with unit and measurement converions
unit_conversions_dict = 'unit_conversions.json'

# top n goods
top_n = 5

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

def extract_goods(file, file_date):
    entities = extract_entities(file)
    all_goods = []
    for sent_id, ents in entities.items():
        goods = []
        previous = [None, None]
        date = file_date

        for type, ent in ents:
            # if type == 'DATE': # if sentence contains date, redefine var. date
            #     date = ent
            if type == 'GOODS' and previous[0] == 'QUANTITY':
                goods.append((date, previous[1], ent))

            previous = [type, ent]

        if goods != []:
            all_goods.extend(goods)

    return all_goods

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
#           EXTRACTING GOODS FROM SYSTEM OUTPUT           #
###########################################################

# :: Dictionary with the date for each document ::
with open(gm_dates) as json_file:
    doc_dates = json.load(json_file)

# :: List of system output files ::
files = glob.glob(system_output_folder + '/*.txt')

# :: Create a table with [date, quantity, goods] ::
error_files = []
goods = []
for file in tqdm(files):
    try:
        filename = file.split('/')[-1].split('.')[0].replace('system-','')
        date = doc_dates[filename]
        file_goods = extract_goods(file, date)
        goods += file_goods
    except Exception as e: # Error tokenizing data. C error: EOF inside string starting at row quotation
        error_files.append(file)

print(f':: No. of files failed to process: {len(error_files)} ::')

# Dataframe with all extracted goods
df_original = pd.DataFrame(goods, columns=['date', 'quantity', 'goods'])

###########################################################
#       CLUSTERING GOODS USING FUZZY STRING MATCHING      #
###########################################################

# :: Clustering goods based on levenshtein similarity score
df = df_original
df['goods_cluster'] = clustering(df['goods'])

# :: Restrict goods to topn most frequently occurring clusters ::
topn = top_n + 5
most_freq_goods = {k:v for k, v in Counter(df['goods_cluster']).most_common(topn)}
df = df[df['goods_cluster'].isin(most_freq_goods)]

print('Most frequently mentioned goods: \n', most_freq_goods)

# Clustering fine-tuning
df = df.replace({'goods_cluster' : {'rijs':'rijst', 'thee boey':'thee', 'zijde stoffen':'zijde', 'gout':'goud', 'thee bing':'thee', 'thee zonglo':'thee', 'thee hayson':'thee', 'fijne thee':'thee', 'peeper':'peper', 'canneel':'kaneel', 'sijde':'zijde'}})

###########################################################
#                      QUANTITIES                         #
###########################################################

# :: Load unit conversion dictionary ::
with open(unit_conversions_dict) as json_file:
    unit_dict = json.load(json_file)
    unit_dict['standardizations'][np.nan] = np.nan

# Split quantities into amounts and units
# Remove quantities without amounts
regex = r'(\d+[.,]*\d*)\s*([A-Za-z]+[.]*[A-Za-z]*)*'
df[['amount','unit']] = df['quantity'].str.extract(regex, expand=True)
df = df.dropna(subset=['amount'])

# Standardize unit spellings
df = df.replace({'unit' : unit_dict['standardizations']})

# Keep only the rows with pre-approved units
df = df[df['unit'].isin(unit_dict['standardizations'].values())]

# Convert amounts and units to uniform measurements
def convert_amount(row):
    if row['unit'] in unit_dict['conversions_amount']:
        conversion_factor = unit_dict['conversions_amount'][row['unit']]
        return int(row['amount'].replace('.','')) * conversion_factor
    else:
        return int(row['amount'].replace('.',''))

df['amount'] = df.apply(convert_amount, axis=1)
df = df.replace({'unit': unit_dict['conversions_unit']})

###########################################################
#                          DATE                           #
###########################################################

# :: Extract year ::
df['year'] = df['date'].str.extract(r'(\d{4})')

###########################################################
#                      FILTER & SAVE                      #
###########################################################

# :: Save table ::
df = df[df['unit'].isin(['lb.'])]
df = df.dropna(subset=['year'])

df.to_csv('timeline_goods_final.tsv', sep='\t')

###########################################################
#                      VISUALIZATION                      #
###########################################################

gm_decades = [1610, 1620, 1630, 1640, 1650, 1660, 1670, 1680, 1690, 1700, 1710, 1720, 1730, 1740, 1750, 1760]

# Read table
df = pd.read_csv('timeline_goods_final.tsv', sep='\t')

# Restrict to goods measured in lb.
df = df[df['unit'].isin(['lb.'])]
df = df.dropna(subset=['year'])

# Restrict to top_n most common goods
topn = top_n
most_freq_goods = {k:v for k, v in Counter(df['goods_cluster']).most_common(topn)}
df = df[df['goods_cluster'].isin(most_freq_goods)]

# Make sure all amounts are numbers
df['amount'] = df['amount'].apply(lambda x: float(x))

# Add decades column
decades = df['year'].apply(lambda x: int(x)//10*10)
df['decade'] = decades

# Group by goods and decade
grouped = df.groupby(['goods_cluster','decade'])['amount'].sum()

# Restructure to dictionary
gm_decades = [1610, 1620, 1630, 1640, 1650, 1660, 1670, 1680,
              1690, 1700, 1710, 1720, 1730, 1740, 1750, 1760]
goods_dict = {}
for k, amount in grouped.to_dict().items():
    goods_type = k[0]
    decade = k[1]

    if goods_type not in goods_dict:
        goods_dict[goods_type] = {k:0 for k in gm_decades}
        goods_dict[goods_type][decade] = amount
    else:
        goods_dict[goods_type][decade] = amount

# Plot timeline
types = list(goods_dict.keys())
# types.remove('rijst')

plt.style.use("ggplot")
plt.figure(figsize=(10,4))
for good in types:
    tuples = goods_dict[good].items()
    x = [i[0] for i in tuples]
    y = [i[1] for i in tuples]
    plt.plot(x, y, marker = '.', linewidth=3)
plt.xlabel('year', fontsize=13)
plt.ylabel('lb.', fontsize=13)
# plt.ticklabel_format(axis="y", style="plain")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(types, loc='best')
plt.show()
