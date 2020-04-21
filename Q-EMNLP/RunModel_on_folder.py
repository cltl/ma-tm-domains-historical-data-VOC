#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_on_folder.py modelPath inputPath
# Usage: python RunModel_on_folder.py models/tmd_annotations_0.7395_0.7185_27.h5 ../gm-txt
# For pretrained models see docs/Pretrained_Models.md

from __future__ import print_function
import glob
import spacy
from spacy.symbols import ORTH
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys
from tqdm import tqdm

if len(sys.argv) < 3:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]

def customize_spacy_tokenizer():
    abbreviations = {'lb.', 'rds.', 'Ned.', 'voorsz.', 'voors.', 'Willemsz.', 'Hoogh.', 'Hoogagtb.', 'septemb.','St.','decemb.','aug.'}

    for token in abbreviations:
        special_case = [{ORTH: token}]
        nlp.tokenizer.add_special_case(token, special_case)

def filter_files(files):
    '''discard notes and uncompleted paragraphs'''
    filtered_files = []
    for file in files:
        if any([file.endswith('_notes.txt'),
                'diff_wrt_uncompleted_paragraphs' in file,
                'tei_ids.txt' in file
                ]):
            continue
        else:
            assert file.endswith('.txt'), f'{file} is not a text file'
            filtered_files.append(file)

    return filtered_files

def tokenize(str):
    doc = nlp(str)
    tokenized = [{'tokens':[i.text for i in sent if '\n' not in i.text]} for sent in doc.sents]
    return tokenized

def run_model(file):
    # :: Read input ::
    with open(file, 'r') as f:
        text = f.read()

    # :: Prepare the input ::
    sentences = tokenize(text)

    addCharInformation(sentences)
    addCasingInformation(sentences)
    dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

    # :: Tag the input ::
    tags = lstmModel.tagSentences(dataMatrix)

    conll = []
    for sentenceIdx in range(len(sentences)):
        tokens = sentences[sentenceIdx]['tokens']

        for tokenIdx in range(len(tokens)):
            tokenTags = []
            for modelName in sorted(tags.keys()):
                tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

            conll.append("%s\t%s\t%s" % (sentenceIdx+1, tokens[tokenIdx], "\t".join(tokenTags)))
        conll.append("")

    conll = "\n".join(conll)

    output_filename = 'system_output/system-' + file.split('/')[-1]
    with open(output_filename, 'w') as outfile:
        outfile.write(conll)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# :: Load the model ::
lstmModel = BiLSTM.loadModel(modelPath)
print(':: Model loaded. ::')

# :: Customize spacy tokenizer ::
nlp = spacy.load('nl_core_news_sm', disable=['ner','tagger'])
nlp.max_length = 1500000
customize_spacy_tokenizer()

# :: List of files to process ::
files = glob.glob(inputPath + '/**/*.txt', recursive=True)
files = filter_files(files)

# :: Run Model on files::
for file in tqdm(files):
    run_model(file)

print(f':: DONE! NER model run on {len(files)} files. ::')
