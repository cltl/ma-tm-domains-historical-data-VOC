#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/

# python RunModel_ConLL_Format.py models/tmd_annotations_0.7395_0.7185_27.h5 data/tmd_annotations/test.txt

from __future__ import print_function
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys
import logging

if len(sys.argv) < 3:
    print("Usage: python RunModel_CoNLL_Format.py modelPath inputPathToConllFile")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]
inputColumns = {1: "tokens"}


# :: Prepare the input ::
sentences = readCoNLL(inputPath, inputColumns)
addCharInformation(sentences)
addCasingInformation(sentences)


# :: Load the model ::
lstmModel = BiLSTM.loadModel(modelPath)


dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
output = []
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']

    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

    #     print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    # print("")

        output.append("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    output.append('')

# Save to file
output = '\n'.join(output)
with open('predictions_on_test_data.txt', 'w') as outfile:
    outfile.write(output)

print(':: DONE! ::')
