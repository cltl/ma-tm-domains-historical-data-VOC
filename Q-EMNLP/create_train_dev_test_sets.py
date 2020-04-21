##### Create train, dev, and test files to train model
##### Convert annotation files to the exact format needed to train model

import glob
import pandas as pd
import numpy as np
from random import shuffle

annotations_folder = '../Annotations/*.tsv'
output_folder = 'data/tmd_annotations'

def resentify(file):
    """
    Convert .tsv file to list of tokenized sentences
    Remove tokens which contain whitespace
    format: [
             [('XXXII', 'O'), ('.', 'O')],
             [('Hebbende', 'O'),('men', 'O'),('wijders', 'O'),('uyt', 'O'), ...]
            ]
    """
    data = pd.read_csv(file, sep='\t') ; len(data)
    data = data[data.sent_id != 'O'] ; len(data)
    data = data.dropna()
    data = data.replace('B-MISC', 'O')
    data = data.replace('I-MISC', 'O')

    # remove tokens that contain whitespaces
    data = data[~data.token.str.contains("\n")]
    data = data[~data.token.str.contains("\t")]
    data = data[~data.token.str.contains(" ")]

    empty = False
    agg_func = lambda s: [(w, t) for w, t in zip(s["token"].values.tolist(),
                                                 s["tag"].values.tolist())]
    grouped = data.groupby("sent_id").apply(agg_func)
    sentences = [s for s in grouped]

    return sentences

def merge_sentences(annotations_folder):
    sentences = []
    for file in glob.glob(annotations_folder):
        sents = resentify(file)
        sentences.extend(sents) # concatenate sentences from multiple files
    print(f":: No. of annotated sentences: {len(sentences)} :: \n")
    return sentences

def conllify(sentences):
    """
    return string in conll format
    """
    # sentences = resentify(file)

    conll = []
    for sent_id, s in enumerate(sentences, 1):
        for token_id, tuple in enumerate(s, 1):
            token = str(tuple[0])
            tag = str(tuple[1])

            # add tab separated conll row
            row = '\t'.join([str(token_id), token, tag])
            conll += [row]

        # add empty line between sentences as per conll convention
        conll += ['']
    conll = '\n'.join(conll)
    return conll

def train_val_test_split(sentences):
    shuffle(sentences)

    n_sents = len(sentences) ; n_sents

    test_sentences  = sentences[ : int(n_sents * 0.1)]
    val_sentences   = sentences[int(n_sents * 0.1) : int(n_sents * 0.2)]
    train_sentences = sentences[int(n_sents * 0.2) : ]

    return train_sentences, val_sentences, test_sentences

def save_train_dev_test_files(sentences, output_folder):
    for set, type in zip(train_val_test_split(sentences),
                             ['train','dev','test']):

        conll = conllify(set)
        n_sents_in_set = len(set)
        filepath = output_folder +'/'+ type + '.txt'

        with open(filepath, 'w') as outfile:
            outfile.write(conll)

        print(f'Created {type} set: {n_sents_in_set} sentences')

### call functions ###
sentences = merge_sentences(annotations_folder)

# Convert sentences into train, dev, and test sets
# save to folder
save_train_dev_test_files(sentences, output_folder)
