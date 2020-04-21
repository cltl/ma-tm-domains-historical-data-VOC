# This script trains the BiLSTM-CNN-CRF architecture for NER in German using
# the GermEval 2014 dataset (https://sites.google.com/site/germeval2014ner/).
# The code use the embeddings by Reimers et al. (https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/)
from __future__ import print_function
import os
import json
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

from keras import backend as K

# :: Change into the working dir of the script ::
# abspath = os.path.abspath(__file__)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

print(':: Logging level done :: \n')

######################################################
#                                                    #
#      D A T A   P R E P R O C E S S I N G           #
#                                                    #
######################################################

datasets = {
    'tmd_annotations':                                   #Name of the dataset
        {'columns': {1:'tokens', 2:'NER_BIO'},    #CoNLL format for the input data. Column 1 contains tokens, column 2 contains NER information using BIO encoding
         'label': 'NER_BIO',                      #Which column we like to predict
         'evaluate': True,                        #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}                   #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}

# :: Path on your computer to the word embeddings. Embeddings by Reimers et al. will be downloaded automatically ::
# embeddingsPath = 'reimers_german_embeddings.gz'
embeddingsPath = 'word_embeddings.txt'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)

print(':: Preprocessing done :: \n')

######################################################
#                                                    #
#         T R A I N I N G   N E T W O R K            #
#                                                    #
######################################################

# Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'],
          'LSTM-Size': [100, 100],
          'dropout': (0.25, 0.25),
          'charEmbeddings': 'CNN',
          'maxCharLength': 50}

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
history = model.fit(epochs=30)

with open('training_history.json', 'w') as outfile:
    json.dump(history, outfile)

# # Visualize training
# import matplotlib.pyplot as plt
# hist = pd.DataFrame(history)
# plt.style.use("ggplot")
# plt.figure(figsize=(7,5))
# plt.plot(hist["epoch"], hist["dev"])
# plt.plot(hist["epoch"], hist["test"])
# plt.xlabel('Epoch')
# plt.ylabel('F1 score')
# plt.ylim(0, 1)
# plt.legend(['dev', 'test'], loc='upper left')
# plt.show()
