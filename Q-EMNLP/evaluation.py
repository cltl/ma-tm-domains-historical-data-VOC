# Generates a training history graph, entity label distribution dictionary, precision, recall and F1 scores, classification report per label, and a confusion matrix

import pandas as pd
import glob
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sn
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

###########################################################
#                      USER INPUTS                        #
###########################################################

# File locations of the gold test data and system generated test data
gold_location = 'data/tmd_annotations/test.txt'
system_location = 'predictions_on_test_data.txt'

# Location train/dev/test files
traindevtest = 'data/tmd_annotations/*'

# Location to training history json
traininghistoryfile = 'training_history.json'

####### Frequencies of IOB tags in corpus #######

files = glob.glob(traindevtest)
df = pd.concat([pd.read_csv(file, sep='\t', names=['sentid','token','tag']) for file in files])

tag_counts = Counter(df['tag'])
tag_counts = {k:v for k,v in tag_counts.most_common(100)}

entity_types = list({i[2:] for i in tag_counts.keys() if i != ('O' or 'text')}) + ['O']
entity_types.remove('text')

print(f':: Entity distribution ::\n{tag_counts} \n')

####### Training History #######

with open(traininghistoryfile) as json_file:
    training_history = json.load(json_file)

# Visualize training
hist = pd.DataFrame(training_history)
plt.style.use("ggplot")
plt.figure(figsize=(4,3))
plt.plot(hist["epoch"], hist["dev"])
plt.plot(hist["epoch"], hist["test"])
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('F1 score', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,1)
plt.legend(['dev', 'test'], loc='lower right')
plt.show()

####### Load gold and system data #######

golddf = pd.read_csv(gold_location, sep='\t', names=['sentid','token','gold']).drop('sentid', axis=1)
systemdf = pd.read_csv(system_location, sep='\t', names=['token','system'])

merged = pd.concat([golddf, systemdf], axis=1) # merge to make sure the tokens are aligned

###########################################################
#                       EVALUATION                        #
###########################################################

# Classification scores
print(':: Classification report ::')
print("Precision score: {:.1%}".format(precision_score(merged['gold'], merged['system'])))
print("Recall score: {:.1%}".format(recall_score(merged['gold'], merged['system'])))
print("F1 score: {:.1%}".format(f1_score(merged['gold'], merged['system'])), '\n')
print(classification_report(merged['gold'], merged['system']))

######## Confusion Matrix ########

def removeIOB(df):
    '''removes B- and I- prefixes from entity labels'''
    new = df
    new = new.replace('B-', '', regex=True)
    new = new.replace('I-', '', regex=True)
    return new

def confusion_matrix(gold, system, entity_types):
    gold = removeIOB(gold)
    system = removeIOB(system)
    confusion_matrix = pd.crosstab(pd.Series(gold),
                                   pd.Series(system),
                                   rownames=['True'],
                                   colnames=['Predicted'],
                                   margins=True,
                                   normalize='index'
                                   )#*100

    df_cm = pd.DataFrame(confusion_matrix,
                         index = entity_types,
                         columns = entity_types)

    sn.set(font_scale=1.2) #for label size
    plt.figure(figsize=(6,4))
    cm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                    cmap="YlGnBu", vmin=0, vmax=1)
    # cm.invert_yaxis()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45)
    # plt.title("Confusion matrix\n")
    plt.show()

confusion_matrix(merged['gold'], merged['system'], entity_types)
