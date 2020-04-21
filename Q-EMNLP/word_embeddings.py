# http://resources.huygens.knaw.nl/retroboeken/memories_ambon/#page=33&accessor=toc&source=1

import spacy
from spacy.symbols import ORTH
# import numpy as np
import pandas as pd
# import re
import glob
import gensim
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
from string import punctuation, whitespace
from gensim.models import KeyedVectors

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def filter_files(files):
    '''discard notes and uncompleted paragraphs'''
    filtered_files = []
    for file in files:
        if any([file.endswith('_notes.txt'),
                'tei_ids.txt' in file,
                'Icon' in file,
                'diff_wrt_uncompleted_paragraphs' in file
                ]):
            continue
        else:
            filtered_files.append(file)

    return filtered_files

def tokenize(file):
    '''file : text file'''

    with open(file, 'r') as infile:
        text = infile.read()
        text = text.lower()
        text = text.replace('\n', ' ')

    doc = nlp(text)
    tokenized = [[i.text for i in sent if '\n' not in i.text] for sent in doc.sents]
    return tokenized

# Add special case rules
def customize_spacy_tokenizer():
    tokens = ['lb.', 'rds.', 'Ned.', 'voorsz.', 'voors.'] # tokens that should not be split

    for token in tokens:
        special_case = [{ORTH: token}]
        nlp.tokenizer.add_special_case(token, special_case)

################ PREPROCESSING #################

# Load SpaCy
nlp = spacy.load('nl_core_news_sm', disable=['ner','tagger'])
customize_spacy_tokenizer()
nlp.max_length = 1500000

# List of usable files
gm = filter_files(glob.glob('gm-txt/*/*'))
gm_uncompleted = glob.glob('gm-text/diff_wrt_uncompleted_paragraphs/*/*')
vandam = filter_files(glob.glob('vandam-txt/*/*'))
extra = glob.glob('extra_texts/*')

files = gm + gm_uncompleted + vandam + extra ; len(files)

# Tokenize text into sentences and word tokens
tokenized_sentences = []
untokenized_files = []
for file in tqdm(files):
    try:
        tokenized = tokenize(file)
    except Exception as e:
        print(file)
        print(e)
        untokenized_sentences += file

    tokenized_sentences += tokenized

print(f'No. of tokenized sentences: {len(tokenized_sentences)}')

tokenized_sentences[:10]

##################### CREATE EMBEDDINGS ######################

# create embeddings
epoch_logger = EpochLogger()
embeddings = Word2Vec(tokenized_sentences,
                      size = 100,
                      window = 5,
                      min_count = 3,
                      iter = 10,
                      callbacks = [epoch_logger])

# most similar words
embeddings.wv.similar_by_word('oock', topn=20)

# save embeddings
embeddings.wv.save_word2vec_format('word_embeddings2.txt', binary=False)

# load embeddings
embeddings = KeyedVectors.load_word2vec_format('word_embeddings.txt', binary=False, unicode_errors='ignore')
