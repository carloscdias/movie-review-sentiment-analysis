#!/bin/env python

from glob import glob
from argparse import *
from collections import defaultdict
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords


# Preprocess text
def preprocess_text (text, options):
    # basic mandatory preprocess: remove html '<br />' and tokenize lower text
    tokens = word_tokenize(text.lower().replace('<br />', ''))

    # Remove stop words optionally
    if options.remove_stopwords:
        sw = stopwords.words('english')
        tokens = [w for w in tokens if w not in sw]

    return tokens


##
## -- EXTRACTORS --
##

# This extractor only counts the presence of the words "good" and "bad"
def dumb_extractor (tokens, dataset):
    dataset['good'].append(tokens.count('good'))
    dataset['bad'].append(tokens.count('bad'))

# Do nothing yet
def smart_extractor (tokens, dataset):
    pass


# This program receives the method extractor and the data directory as params
def main():
    extract_modes = {
        'dumb': dumb_extractor
    }

    options = list(extract_modes.keys())

    # here data will be stored
    dataset = defaultdict(list)

    parser = ArgumentParser(description = "This program extract features based in a given previous defined method using data found in the given data directory")

    parser.add_argument("method", choices = options, help = "Method to be used to extract data")
    parser.add_argument("data_directory", help = "Directory (whitout the last trailing slash) from which data will be read")

    parser.add_argument("-p", "--positive-label", default = "positive", help = "Label to be used in positive use cases")
    parser.add_argument("-n", "--negative-label", default = "negative", help = "Label to be used in negative use cases")

    parser.add_argument("-r", "--remove-stopwords", action = 'store_true', help = "Remove common words in the englis vocabulary before processing")

    args = parser.parse_args()

    # Read examples and parse features
    for type in ["pos", "neg"]:
        for file in glob("{}/{}/*.txt".format(args.data_directory, type)):
            # Read file and preprocess into text variable
            with open(file) as f:
                tokens = preprocess_text (f.read(), args)

            # Extract features with the given method
            extract_modes[args.method] (tokens, dataset)
            # Put label
            dataset['label'].append (args.positive_label if type == 'pos' else args.negative_label)

    # Data set to data frame
    df = pd.DataFrame(dataset)

    # Data preview
    print("\nPreview")
    print(df.head())

    # describe data
    print("\nStatistics")
    print(df.describe())

    # save data
    file_name = "{}/{}.csv".format(args.data_directory, args.method)
    print ("\n\nSaving dataset into '{}'".format(file_name))

    # saving...
    df.to_csv(file_name, index = False)


if __name__ == '__main__':
    main()

