#!/bin/env python

from glob import glob
from argparse import *
from collections import defaultdict
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
import string

## New stuff!
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocess text
def preprocess_text (text, options):
    # basic mandatory preprocess: remove html '<br />' and tokenize lower text removing punctuatuion
    table = str.maketrans({ch: None for ch in string.punctuation})
    tokens = [w.translate(table) for w in word_tokenize(text.lower().replace('<br />', ''))]

    # Remove stop words optionally
    if options.remove_stopwords:
        sw = stopwords.words('english')
        tokens = [w for w in tokens if w not in sw]

    if options.remove_one_letter:
        tokens = [w for w in tokens if len(w) > 1]

    # Stem words
    if options.stemmer:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


##
## -- EXTRACTORS --
##

# This extractor only counts the presence of the words "good" and "bad"
def dumb_extractor (tokens, dataset):
    for t in tokens:
        dataset['good'].append(t.count('good'))
        dataset['bad'].append(t.count('bad'))

# Do nothing yet
def unigram_extractor (tokens, dataset):
    # create corpus
    token_set = set()
    for e in tokens:
        token_set.update(e)
    
    # count elements for each row
    for token in tokens:
        for unit in token_set:
            dataset[unit].append(token.count(unit))


# This program receives the method extractor and the data directory as params
def main():
    extract_modes = {
        'dumb': dumb_extractor,
        'unigram': unigram_extractor
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
    parser.add_argument("-s", "--stemmer", action = 'store_true', help = "Use Porter Stemmer to stem words")
    parser.add_argument("-o", "--remove-one-letter", action = 'store_true', help = "Remove one letter words")

    args = parser.parse_args()

    vectorizer = TfidfVectorizer(input = 'filename', ngram_range = (1, 1))

    # load data
    positive_data = glob("{}/pos/*.txt".format(args.data_directory))
    negative_data = glob("{}/neg/*.txt".format(args.data_directory))
    # data length
    len_positive = len(positive_data)
    len_negative = len(negative_data)

    # example labels
    y = ([args.positive_label] * len_positive) + ([args.negative_label] * len_negative)

    t = SelectKBest()
    b = t.fit(result, y)
    b.get_support() # mask array
    res = tfidf.fit_transform(text, y)
    np.array(res.get_feature_names())

    # Read examples and parse features
    for type in ["pos", "neg"]:
        for file in glob("{}/{}/*.txt".format(args.data_directory, type)):
            # Read file into text array
            with open(file) as f:
                text.append(f.read())

            # Put label
            dataset['_result_label'].append (args.positive_label if type == 'pos' else args.negative_label)

    # Extract features with the given method
    tokens = []
    for t in text:
        tokens.append(preprocess_text (t, args))

    extract_modes[args.method] (tokens, dataset)

    for i in dataset.keys():
        print("{} -> {}".format(i, len(dataset[i])))

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

